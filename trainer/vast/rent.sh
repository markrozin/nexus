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
BUNDLE="$ROOT/dist/newchessbot-train.tar.gz"
STATE="$ROOT/dist/vast-session"
NETS_OUT="$ROOT/nets/$MODE-$(date +%Y%m%d-%H%M)"

MAX_PRICE="${MAX_PRICE:-0.70}"
MAX_DOLLARS="${MAX_DOLLARS:-3}"
CREDIT_FLOOR="${CREDIT_FLOOR:-2}"
BOOT_TIMEOUT_MIN="${BOOT_TIMEOUT_MIN:-20}"
POLL_SECONDS=120
IMAGE="vastai/base-image:cuda-12.8.1-auto"
DISK_GB=100
LABEL="newchessbot-$MODE"
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
        | json 'rows = d if isinstance(d, list) else d.get("instances", []); print(" ".join(str(r["id"]) for r in rows))'
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

ssh_run() { ssh "${SSH_OPTS[@]}" "$@"; }

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

CAP_MINUTES=$("$PY" -c "print(int($MAX_DOLLARS / $PRICE * 60))")
log "limits: \$$MAX_DOLLARS cap at \$$PRICE/hr = $CAP_MINUTES minutes; local deadline adds 20"

if [ "${DRY_RUN:-0}" = 1 ]; then
    log "DRY_RUN=1: stopping before renting anything"
    exit 0
fi

# ---------------------------------------------------------------------------
log "creating instance from offer $OFFER"
vast create instance "$OFFER" --image "$IMAGE" --disk "$DISK_GB" --ssh --direct \
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

url=$(vast ssh-url "$INSTANCE" 2>/dev/null | tr -d '\r')
[[ "$url" =~ ssh://([^@]+)@([^:]+):([0-9]+) ]] || die "unexpected ssh-url output: $url"
SSH_USER="${BASH_REMATCH[1]}"
SSH_HOST="${BASH_REMATCH[2]}"
SSH_PORT="${BASH_REMATCH[3]}"
SSH_OPTS=(-p "$SSH_PORT" -i "$HOME/.ssh/id_ed25519" -o StrictHostKeyChecking=accept-new
    -o UserKnownHostsFile="$STATE/known_hosts" -o ConnectTimeout=20 -o ServerAliveInterval=30
    -o BatchMode=yes "$SSH_USER@$SSH_HOST")
log "ssh $SSH_USER@$SSH_HOST:$SSH_PORT"

until ssh_run true 2> "$STATE/ssh.err"; do
    [ "$(date +%s)" -lt "$boot_deadline" ] || die "ssh never became reachable: $(tail -2 "$STATE/ssh.err")"
    sleep 15
done
log "ssh reachable"

# ---------------------------------------------------------------------------
log "uploading the bundle"
scp -P "$SSH_PORT" -i "$HOME/.ssh/id_ed25519" -o StrictHostKeyChecking=accept-new \
    -o UserKnownHostsFile="$STATE/known_hosts" -o BatchMode=yes \
    "$BUNDLE" "$SSH_USER@$SSH_HOST:/workspace/newchessbot-train.tar.gz" || die "upload failed"
local_sum=$(sha256sum "$BUNDLE" | cut -d' ' -f1)
remote_sum=$(ssh_run "sha256sum /workspace/newchessbot-train.tar.gz" | cut -d' ' -f1)
[ "$local_sum" = "$remote_sum" ] || die "bundle checksum mismatch after upload"
log "bundle verified on the instance"

# The container's own environment (CONTAINER_ID, CONTAINER_API_KEY) is not
# guaranteed in an ssh session, so it is read from PID 1 for run.sh's
# self-destroy.
ssh_run "cd /workspace && tar -xzf newchessbot-train.tar.gz && cd newchessbot-train \
    && export \$(tr '\\0' '\\n' < /proc/1/environ | grep -E '^(CONTAINER_ID|CONTAINER_API_KEY)=' | xargs) \
    && PRICE_PER_HOUR=$PRICE MAX_DOLLARS=$MAX_DOLLARS nohup bash trainer/vast/run.sh $MODE > run.log 2>&1 < /dev/null &" \
    || die "could not start run.sh"
log "run.sh $MODE started"

# ---------------------------------------------------------------------------
last_stage=""
unreachable=0
while :; do
    past_deadline && die "local wall-clock deadline reached"
    if ssh_run "cat /workspace/newchessbot-train/run.log" > "$STATE/run.log" 2> "$STATE/ssh.err"; then
        unreachable=0
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
            for path in $(grep -A10 '^final networks:' "$STATE/run.log" | grep -oE '/workspace/[^ ]+\.bin'); do
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
