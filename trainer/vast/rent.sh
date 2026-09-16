#!/usr/bin/env bash
# Rent a vast.ai GPU through the API, run one guarded training session, bring
# the network home, and make sure the instance is dead.
#
#   bash trainer/vast/rent.sh lichess          # the real thing
#   DRY_RUN=1 bash trainer/vast/rent.sh lichess  # preflight and offer choice only
#
# Needs the vastai CLI with an API key already set (`vastai set api-key`, done
# by a human, never by this script), a registered SSH key, and a bundle from
# `prepare.sh --lichess` that printed PREFLIGHT PASSED.
#
# STRICT LIMITS -- each one refuses or kills rather than warns:
#
#   - One instance at a time. If the account already has ANY instance, this
#     refuses to start. `reap.sh` destroys stragglers.
#   - Price ceiling: no offer above MAX_PRICE $/hr (all-in, with the disk).
#   - Spend ceiling: the session may not cost more than MAX_DOLLARS, and the
#     account must keep CREDIT_FLOOR dollars after that worst case.
#   - Boot deadline: an instance not running and reachable within
#     BOOT_TIMEOUT_MIN minutes is destroyed.
#   - Wall-clock deadline: whatever happens, the instance is destroyed once
#     MAX_DOLLARS / price hours (plus a small margin) have passed.
#   - Three independent killers: this script's EXIT trap, run.sh's own trap on
#     the instance, and run.sh's detached hard deadline. Any one suffices.
#
# "Dead" means verified, not requested: after every destroy the account's
# instance list is re-read until the id is gone, retrying the destroy each
# time. Failure to verify is loud, leaves dist/VAST_INSTANCE_NOT_DEAD behind,
# and exits non-zero.

set -uo pipefail # deliberately not -e: the destroy path must run after any failure

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
V="${VASTAI:-/c/Users/Mark/AppData/Roaming/Python/Python314/Scripts/vastai.exe}"
PY="${PYTHON:-python}"
MODE="${1:-lichess}"
BUNDLE="$ROOT/dist/nexus-train.tar.gz"
STATE="$ROOT/dist/vast-session"
NETS_OUT="$ROOT/nets/$MODE-$(date +%Y%m%d-%H%M)"

MAX_PRICE="${MAX_PRICE:-0.70}"
MAX_DOLLARS="${MAX_DOLLARS:-3}"
CREDIT_FLOOR="${CREDIT_FLOOR:-2}"
BOOT_TIMEOUT_MIN="${BOOT_TIMEOUT_MIN:-12}"
POLL_SECONDS=120
# Instances are created from vast.ai's own NVIDIA CUDA template, not a bare
# image: the base image starts sshd only through the template's onstart
# (entrypoint.sh). Session 3 created from the bare image and its ssh port
# refused every connection for 20 minutes. The newest official version is
# looked up at run time.
TEMPLATE_QUERY='name="NVIDIA CUDA" creator_id=62897'
DISK_GB=100
LABEL="nexus-$MODE"
# China is excluded: Hugging Face is unreliable from there, and the session
# downloads its data from Hugging Face.
QUERY="gpu_name=RTX_4090 num_gpus=1 verified=true rentable=true reliability>0.98 inet_down>=500 cuda_vers>=12.8 disk_space>=$DISK_GB cpu_cores_effective>=12 direct_port_count>=1 geolocation notin [CN]"

INSTANCE=""
SSH_OPTS=()

mkdir -p "$STATE"
log() { printf '%s %s\n' "$(date +%T)" "$*"; }
die() { log "FATAL: $*"; exit 1; }
# Run python over JSON on stdin; the expression sees it as `d`.
json() { "$PY" -c "import json, sys; d = json.load(sys.stdin); $1"; }

vast() { "$V" "$@"; }

# Every instance id on the account, space-separated. Fails (non-zero) if the
# list cannot be read -- which must never be mistaken for "no instances".
list_ids() {
    vast show instances --raw 2> "$STATE/list.err" \
        | json 'rows = d if isinstance(d, list) else d.get("instances", []); print(" ".join(str(r["id"]) for r in rows))' 2>/dev/null
}

destroy_verified() {
    local id="$1" attempt ids
    for attempt in $(seq 1 30); do
        vast destroy instance "$id" -y > "$STATE/destroy.$attempt.log" 2>&1
        sleep 20
        if ! ids=$(list_ids); then
            log "could not read the instance list (attempt $attempt); retrying"
            continue
        fi
        case " $ids " in
            *" $id "*) log "instance $id still listed after destroy attempt $attempt" ;;
            *)
                log "VERIFIED DEAD: instance $id is no longer on the account"
                [ -z "$ids" ] || log "WARNING: other instances exist on the account: $ids"
                rm -f "$STATE/instance.id"
                return 0
                ;;
        esac
    done
    log "!!! INSTANCE $id COULD NOT BE VERIFIED DEAD AFTER 30 ATTEMPTS -- DESTROY IT IN THE CONSOLE NOW !!!"
    touch "$ROOT/dist/VAST_INSTANCE_NOT_DEAD"
    return 1
}

on_exit() {
    local code=$?
    trap '' INT TERM HUP
    if [ -n "$INSTANCE" ]; then
        log "session over (exit code $code): destroying instance $INSTANCE"
        destroy_verified "$INSTANCE" || code=9
        log "credit now: $(vast show user --raw 2>/dev/null | json 'print(round(d["credit"], 2))' 2>/dev/null)"
    fi
    exit "$code"
}
trap on_exit EXIT
trap 'exit 130' INT TERM HUP

# Every remote command has a timeout and no stdin: a hung connection must fail,
# not stall the controller while the instance bills.
ssh_run() { timeout 120 ssh -n "${SSH_OPTS[@]}" "$@"; }

# ---------------------------------------------------------------------------
log "preflight"
[ -x "$V" ] || die "vastai CLI not found at $V"
[ -f "$BUNDLE" ] || die "missing $BUNDLE -- run prepare.sh first"
[ -f "$HOME/.ssh/id_ed25519" ] || die "missing ~/.ssh/id_ed25519"
[ ! -e "$ROOT/dist/VAST_INSTANCE_NOT_DEAD" ] || die "a previous instance was never verified dead; check the console and delete dist/VAST_INSTANCE_NOT_DEAD"

credit=$(vast show user --raw 2> "$STATE/user.err" | json 'print(d["credit"])') \
    || die "cannot read the account -- is the API key set?"
existing=$(list_ids) || die "cannot read the instance list"
[ -z "$existing" ] || die "the account already has instances ($existing); one at a time -- run reap.sh"
"$PY" -c "import sys; sys.exit(0 if $credit >= $MAX_DOLLARS + $CREDIT_FLOOR else 1)" \
    || die "credit \$$credit cannot cover MAX_DOLLARS=\$$MAX_DOLLARS plus the \$$CREDIT_FLOOR floor"
log "credit \$$credit, no existing instances, bundle $(sha256sum "$BUNDLE" | cut -c1-16)..."

log "searching offers"
vast search offers "$QUERY" --storage "$DISK_GB" -o dph_total --limit 20 --raw > "$STATE/offers.json" 2> "$STATE/offers.err" \
    || die "offer search failed: $(cat "$STATE/offers.err")"
choice=$(json "
rows = [r for r in d if r.get('dph_total', 99) <= $MAX_PRICE]
if not rows: sys.exit(1)
r = rows[0]
print(r['id'], round(r['dph_total'], 4), r.get('geolocation', '?').replace(' ', '_'), r.get('machine_id'), round(r.get('reliability', 0), 4), int(r.get('inet_down', 0)), r.get('cpu_cores_effective'))
" < "$STATE/offers.json") || die "no offer at or below \$$MAX_PRICE/hr matches the filters"
read -r OFFER PRICE WHERE MACHINE RELIABILITY DOWN CPUS <<< "$choice"
log "chosen offer $OFFER: \$$PRICE/hr, $WHERE, machine $MACHINE, reliability $RELIABILITY, ${DOWN} Mbps down, $CPUS cpus"

TEMPLATE=$(vast search templates "$TEMPLATE_QUERY" --raw 2> "$STATE/templates.err" | json '
rows = d if isinstance(d, list) else d.get("templates", [])
rows = [r for r in rows if r.get("onstart") == "entrypoint.sh" and r.get("use_ssh")]
if not rows: sys.exit(1)
print(max(rows, key=lambda r: r.get("created_at") or 0)["hash_id"])
') || die "cannot find the official NVIDIA CUDA template"
log "template $TEMPLATE (official NVIDIA CUDA, ssh enabled, onstart entrypoint.sh)"

CAP_MINUTES=$("$PY" -c "print(int($MAX_DOLLARS / $PRICE * 60))")
log "limits: \$$MAX_DOLLARS cap at \$$PRICE/hr = $CAP_MINUTES minutes; local deadline adds 20"

if [ "${DRY_RUN:-0}" = 1 ]; then
    log "DRY_RUN=1: stopping before renting anything"
    exit 0
fi

# ---------------------------------------------------------------------------
log "creating instance from offer $OFFER"
vast create instance "$OFFER" --template_hash "$TEMPLATE" --disk "$DISK_GB" \
    --label "$LABEL" --cancel-unavail --raw > "$STATE/create.json" 2> "$STATE/create.err"
INSTANCE=$(json 'print(d["new_contract"]) if d.get("success") else sys.exit(1)' < "$STATE/create.json" 2>/dev/null) \
    || { INSTANCE=""; die "create failed: $(cat "$STATE/create.json" "$STATE/create.err" 2>/dev/null)"; }
echo "$INSTANCE" > "$STATE/instance.id"
START_EPOCH=$(date +%s)
DEADLINE_EPOCH=$((START_EPOCH + (CAP_MINUTES + 20) * 60))
log "instance $INSTANCE created; it is destroyed no later than $(date -d "@$DEADLINE_EPOCH" +%T)"

past_deadline() { [ "$(date +%s)" -ge "$DEADLINE_EPOCH" ]; }

log "waiting for it to run (up to $BOOT_TIMEOUT_MIN minutes)"
boot_deadline=$((START_EPOCH + BOOT_TIMEOUT_MIN * 60))
while :; do
    status=$(vast show instance "$INSTANCE" --raw 2>/dev/null | json 'print(d.get("actual_status"))' 2>/dev/null)
    [ "$status" = running ] && break
    [ "$(date +%s)" -lt "$boot_deadline" ] || die "instance never reached running (last status: ${status:-unknown})"
    sleep 15
done
log "running"

# Every way in the instance data offers: the CLI's ssh-url, the proxy
# (ssh_host:ssh_port) and the direct mapping of container port 22. Each is
# tried in turn until one answers, since which works depends on the machine.
ssh_candidates() {
    vast ssh-url "$INSTANCE" 2>/dev/null | tr -d '\r' | sed -nE 's#^ssh://[^@]+@([^:]+):([0-9]+).*#\1 \2#p'
    vast show instance "$INSTANCE" --raw 2>/dev/null | json '
if d.get("ssh_host") and d.get("ssh_port"): print(d["ssh_host"], d["ssh_port"])
m = (d.get("ports") or {}).get("22/tcp") or []
if d.get("public_ipaddr") and m: print(d["public_ipaddr"].strip(), m[0]["HostPort"])
' 2>/dev/null
}

SSH_HOST=""
until [ -n "$SSH_HOST" ]; do
    while read -r host port; do
        [ -n "$host" ] || continue
        if ssh -p "$port" -i "$HOME/.ssh/id_ed25519" -o StrictHostKeyChecking=accept-new \
            -o UserKnownHostsFile="$STATE/known_hosts" -o ConnectTimeout=15 -o BatchMode=yes \
            "root@$host" true 2>> "$STATE/ssh.err"; then
            SSH_HOST="$host"
            SSH_PORT="$port"
            break
        fi
    done < <(ssh_candidates | sort -u)
    if [ -z "$SSH_HOST" ]; then
        if [ "$(date +%s)" -ge "$boot_deadline" ]; then
            # Keep what the instance reported, minus anything secret, for the post-mortem.
            vast show instance "$INSTANCE" --raw 2>/dev/null | json '
for k in list(d):
    if any(s in k.lower() for s in ("key", "token", "pass", "secret")): d.pop(k)
print(json.dumps(d, indent=1))' > "$STATE/instance-at-failure.json" 2>/dev/null
            die "ssh never became reachable on any endpoint: $(tail -3 "$STATE/ssh.err")"
        fi
        sleep 15
    fi
done
SSH_USER=root
SSH_OPTS=(-p "$SSH_PORT" -i "$HOME/.ssh/id_ed25519" -o StrictHostKeyChecking=accept-new
    -o UserKnownHostsFile="$STATE/known_hosts" -o ConnectTimeout=20 -o ServerAliveInterval=30
    -o BatchMode=yes "$SSH_USER@$SSH_HOST")
log "ssh reachable at $SSH_HOST:$SSH_PORT"

# ---------------------------------------------------------------------------
log "uploading the bundle"
scp -P "$SSH_PORT" -i "$HOME/.ssh/id_ed25519" -o StrictHostKeyChecking=accept-new \
    -o UserKnownHostsFile="$STATE/known_hosts" -o BatchMode=yes \
    "$BUNDLE" "$SSH_USER@$SSH_HOST:/workspace/nexus-train.tar.gz" || die "upload failed"
local_sum=$(sha256sum "$BUNDLE" | cut -d' ' -f1)
remote_sum=$(ssh_run "sha256sum /workspace/nexus-train.tar.gz" | cut -d' ' -f1)
[ "$local_sum" = "$remote_sum" ] || die "bundle checksum mismatch after upload"
log "bundle verified on the instance"

ssh_run "cd /workspace && tar -xzf nexus-train.tar.gz && test -f nexus-train/trainer/vast/run.sh" \
    || die "could not unpack the bundle on the instance"

# Session 4 lost 50 minutes here: `a && b && nohup c &` backgrounds the whole
# list in a subshell that still holds ssh's output, so ssh waited for the
# training run to end. Now a launcher script is uploaded and started as one
# detached process with every stream redirected, so ssh returns at once.
cat > "$STATE/launch.sh" <<'EOF'
#!/usr/bin/env bash
# Started detached by rent.sh. The container's environment is not guaranteed
# in an ssh session, so run.sh's self-destroy credentials come from PID 1.
export $(tr '\0' '\n' < /proc/1/environ | grep -E '^(CONTAINER_ID|CONTAINER_API_KEY)=' | xargs)
cd /workspace/nexus-train || exit 1
export PRICE_PER_HOUR=__PRICE__ MAX_DOLLARS=__MAX_DOLLARS__
exec bash trainer/vast/run.sh __MODE__
EOF
sed -i "s/__PRICE__/$PRICE/; s/__MAX_DOLLARS__/$MAX_DOLLARS/; s/__MODE__/$MODE/" "$STATE/launch.sh"
timeout 60 ssh "${SSH_OPTS[@]}" "cat > /workspace/launch.sh" < "$STATE/launch.sh" \
    || die "could not upload the launcher"
timeout 60 ssh -n "${SSH_OPTS[@]}" \
    "setsid nohup bash /workspace/launch.sh > /workspace/nexus-train/run.log 2>&1 < /dev/null & echo launched" \
    || die "could not launch run.sh"

started=0
for _ in $(seq 1 12); do
    sleep 10
    if ssh_run "grep -q '=== checking GPU' /workspace/nexus-train/run.log" 2>/dev/null; then
        started=1
        break
    fi
done
[ "$started" = 1 ] \
    || die "run.sh did not start: $(ssh_run 'tail -5 /workspace/nexus-train/run.log' 2>&1)"
log "run.sh $MODE started"

# ---------------------------------------------------------------------------
last_stage=""
unreachable=0
# A run.log that stops changing means the session is stuck, whatever stage it
# claims. Session 5 sat silent for six hours before its cap destroyed it.
STALE_MINUTES="${STALE_MINUTES:-25}"
last_change=$(date +%s)
last_size=""
while :; do
    past_deadline && die "local wall-clock deadline reached"
    # Into a temp file first: a failed poll must not truncate the last good copy,
    # which is the only diagnostic left once the instance is gone.
    if ssh_run "cat /workspace/nexus-train/run.log" > "$STATE/run.log.tmp" 2> "$STATE/ssh.err" \
        && mv "$STATE/run.log.tmp" "$STATE/run.log"; then
        unreachable=0
        size_now=$(stat -c %s "$STATE/run.log")
        if [ "$size_now" != "$last_size" ]; then
            last_size=$size_now
            last_change=$(date +%s)
        fi
        [ $(($(date +%s) - last_change)) -lt $((STALE_MINUTES * 60)) ] \
            || die "run.log has not changed for $STALE_MINUTES minutes; the session is stuck"
        stage=$(grep -E '^=== ' "$STATE/run.log" | tail -1)
        if [ "$stage" != "$last_stage" ]; then
            log "$stage"
            last_stage="$stage"
        fi
        if grep -q '^SUCCESS' "$STATE/run.log"; then
            log "SUCCESS on the instance -- fetching networks"
            mkdir -p "$NETS_OUT"
            cp "$STATE/run.log" "$NETS_OUT/run.log"
            fetched=0
            # Only the indented lines directly under "final networks:" -- the
            # session-ending listing that follows names every checkpoint file.
            for path in $(awk '/^final networks:/ { f = 1; next } f && /^  \/workspace\// { print $1; next } f { exit }' "$STATE/run.log"); do
                name=$(echo "$path" | awk -F/ '{ print $(NF-1) }')
                if scp -P "$SSH_PORT" -i "$HOME/.ssh/id_ed25519" -o UserKnownHostsFile="$STATE/known_hosts" \
                    -o BatchMode=yes "$SSH_USER@$SSH_HOST:$path" "$NETS_OUT/$name.bin"; then
                    remote=$(ssh_run "sha256sum $path" | cut -d' ' -f1)
                    here=$(sha256sum "$NETS_OUT/$name.bin" | cut -d' ' -f1)
                    if [ "$remote" = "$here" ]; then
                        log "fetched $name.bin ($(stat -c %s "$NETS_OUT/$name.bin") bytes, sha256 verified)"
                        fetched=$((fetched + 1))
                    else
                        log "checksum mismatch on $name.bin"
                    fi
                fi
            done
            [ "$fetched" -gt 0 ] || die "SUCCESS reported but no network could be fetched"
            log "networks in $NETS_OUT"
            exit 0
        fi
        if grep -q '^FAILED' "$STATE/run.log"; then
            cp "$STATE/run.log" "$STATE/failed-run.log"
            tail -30 "$STATE/run.log"
            die "run.sh reported FAILED"
        fi
    else
        unreachable=$((unreachable + 1))
        log "ssh poll failed ($unreachable in a row)"
        # run.sh may already have destroyed the instance itself.
        if ids=$(list_ids); then
            case " $ids " in
                *" $INSTANCE "*) ;;
                *)
                    log "instance $INSTANCE is already gone (self-destroyed)"
                    INSTANCE=""
                    rm -f "$STATE/instance.id"
                    exit 1
                    ;;
            esac
        fi
        [ "$unreachable" -lt 10 ] || die "instance unreachable for 10 polls"
    fi
    sleep "$POLL_SECONDS"
done
