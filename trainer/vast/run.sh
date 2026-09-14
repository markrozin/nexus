#!/usr/bin/env bash
# Train the newchessbot NNUE on a rented vast.ai GPU, with guardrails so money
# is not wasted.
#
# From the bundle root, after uploading it:
#
#   export PRICE_PER_HOUR=0.30   # this instance's price, from the vast.ai listing
#   export MAX_DOLLARS=3         # hard cap on instance time for this session
#   nohup bash trainer/vast/run.sh all > run.log 2>&1 &
#   tail -f run.log
#
#   bash trainer/vast/run.sh check   # only the fast checks, nothing billed-heavy
#
#   nohup bash trainer/vast/run.sh lichess > run.log 2>&1 &
#       The same guarded session, but after the smoke train it streams the
#       Lichess evaluation database (22 GB, CC0) straight into lichesseval on
#       this box -- nothing of the download is stored -- and trains on that at
#       WDL 0.0. Needs ~60 GB free disk; MAX_POSITIONS stops the stream early.
#
# `all` runs: fail-fast checks -> build -> smoke train -> on-box netcheck ->
# real training -> final netcheck -> download window -> the instance destroys
# itself. `nohup` matters: a dropped SSH session must not kill the run halfway.
#
# WHAT THIS CAN AND CANNOT PROMISE
#
# vast.ai bills an instance from the moment it starts until it is DESTROYED.
# Setup is billed exactly like training, a stopped instance still bills for
# disk, and bandwidth is billed per byte in every state. No script can make
# setup free. What these guardrails guarantee instead is that time not spent
# training is short, capped, and impossible to forget about:
#
#   1. Fail fast. The GPU, the CUDA libraries bullet links, the self-destroy key
#      and the budget are all checked before anything slow happens.
#   2. Hard cap. Instance time is capped at MAX_DOLLARS / PRICE_PER_HOUR hours,
#      enforced with `timeout`, with the download window reserved inside the cap.
#   3. Self-destroy. On ANY exit -- success, failure, cap reached, or a signal --
#      the instance destroys itself after a download window. Destroy, never
#      stop. The only opt-out is deliberate: `touch /tmp/KEEP_INSTANCE`.
#   4. Validate before the expensive step. A smoke train plus an on-box
#      netcheck must pass before the real training run starts.
#
# The cap counts from when this script starts. Boot and upload time before that
# is billed but not counted, so start the script promptly.

set -euo pipefail

BUNDLE="$(cd "$(dirname "$0")/../.." && pwd)"
TRAINER="$BUNDLE/trainer"
ENGINE="$BUNDLE/engine"
DATA="$BUNDLE/data"
NETS="$BUNDLE/nets"

SMOKE_DATA="$DATA/tp8.shuffled.data"
SMOKE_TEXT="$DATA/tp8.dedup.txt"
TRAIN_DATA="$DATA/pipeline.shuffled.data"
TRAIN_TEXT="$DATA/pipeline.check.txt"

LICHESS_URL="https://database.lichess.org/lichess_db_eval.jsonl.zst"
# Must equal the rev in trainer/Cargo.toml, so bullet-utils writes exactly the
# format the trainer reads. prepare.sh checks the two agree.
BULLET_REV="629ee50000b2afb7b3337595401c830d3b1e0f42"
# About 230M kept positions: ~16 GB of text, then 7 GB of bulletformat twice
# while shuffling, with headroom for the build.
MIN_FREE_GB_LICHESS=60

DOWNLOAD_GRACE_MIN="${DOWNLOAD_GRACE_MIN:-15}"
FAILURE_GRACE_MIN="${FAILURE_GRACE_MIN:-5}"
EPOCHS="${EPOCHS:-40}"

log() { printf '\n=== %s ===\n' "$*"; }

# ---------------------------------------------------------------------------
# Fail-fast checks. Nothing here costs more than a few seconds.
# ---------------------------------------------------------------------------

need_cuda() {
    log "checking GPU and CUDA toolkit"
    export CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
    if [ ! -d "$CUDA_PATH/lib64" ]; then
        echo "CUDA_PATH=$CUDA_PATH has no lib64: this image lacks the CUDA toolkit." >&2
        echo "Destroy it and pick an nvidia/cuda *-devel* image." >&2
        exit 1
    fi
    # bullet's build script links exactly these from $CUDA_PATH/lib64.
    for lib in libcudart libnvrtc libcublas; do
        if ! ls "$CUDA_PATH"/lib64/"$lib".so* >/dev/null 2>&1; then
            echo "missing $lib in $CUDA_PATH/lib64 -- bullet will not link." >&2
            exit 1
        fi
    done
    export LD_LIBRARY_PATH="$CUDA_PATH/lib64:${LD_LIBRARY_PATH:-}"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
}

need_self_destroy() {
    log "checking the self-destroy guardrail"
    if [ "${NO_AUTODESTROY:-0}" = 1 ]; then
        echo "NO_AUTODESTROY=1: auto-destroy disabled by request."
        echo "YOU must destroy this instance by hand. It bills until you do."
        return
    fi
    if [ -z "${CONTAINER_ID:-}" ] || [ -z "${CONTAINER_API_KEY:-}" ]; then
        echo "CONTAINER_ID or CONTAINER_API_KEY is not set, so this instance cannot" >&2
        echo "destroy itself. Refusing to run without that guardrail." >&2
        echo "Set NO_AUTODESTROY=1 only if you will destroy it by hand." >&2
        exit 1
    fi
    command -v curl >/dev/null 2>&1 || { echo "curl is missing" >&2; exit 1; }
    echo "instance $CONTAINER_ID can destroy itself (key is scoped to this instance only)"
}

need_budget() {
    log "checking the spending cap"
    if [ -z "${PRICE_PER_HOUR:-}" ]; then
        echo "Set PRICE_PER_HOUR to this instance's hourly price, e.g. export PRICE_PER_HOUR=0.30" >&2
        exit 1
    fi
    MAX_DOLLARS="${MAX_DOLLARS:-3}"
    CAP_SECONDS=$(awk -v m="$MAX_DOLLARS" -v p="$PRICE_PER_HOUR" \
        'BEGIN { if (p <= 0 || m <= 0) exit 1; printf "%d", m / p * 3600 }') || {
        echo "PRICE_PER_HOUR and MAX_DOLLARS must both be positive numbers" >&2
        exit 1
    }
    local reserve=$((DOWNLOAD_GRACE_MIN * 60))
    if [ "$CAP_SECONDS" -le $((reserve + 600)) ]; then
        echo "A cap of \$$MAX_DOLLARS at \$$PRICE_PER_HOUR/hr is $((CAP_SECONDS / 60)) minutes," >&2
        echo "too little to build, train and leave a download window. Raise MAX_DOLLARS." >&2
        exit 1
    fi
    echo "cap: \$$MAX_DOLLARS at \$$PRICE_PER_HOUR/hr = $((CAP_SECONDS / 60)) minutes of instance time"
    echo "     ($DOWNLOAD_GRACE_MIN of them reserved for downloading the network)"
}

# Seconds of budget left for work, after reserving the download window.
remaining_seconds() {
    echo $((CAP_SECONDS - SECONDS - DOWNLOAD_GRACE_MIN * 60))
}

# Run a command inside whatever budget is left. Exits (and so self-destroys)
# rather than start work the cap cannot pay for.
with_cap() {
    local left
    left=$(remaining_seconds)
    if [ "$left" -le 60 ]; then
        echo "SPENDING CAP REACHED: not starting: $*" >&2
        exit 3
    fi
    local code=0
    timeout --signal=INT --kill-after=60 "$left" "$@" || code=$?
    if [ "$code" -ne 0 ]; then
        [ "$code" -eq 124 ] && echo "SPENDING CAP REACHED during: $*" >&2
        exit "$code"
    fi
}

# ---------------------------------------------------------------------------
# Self-destroy. Armed only once the checks above prove it can work.
# ---------------------------------------------------------------------------

destroy_instance() {
    if [ "${NO_AUTODESTROY:-0}" = 1 ]; then
        echo "NO_AUTODESTROY=1: NOT destroying. This instance is still billing -- destroy it now."
        return
    fi
    if [ -e /tmp/KEEP_INSTANCE ]; then
        echo "/tmp/KEEP_INSTANCE exists: NOT destroying, by request."
        echo "This instance is still billing. Destroy it in the console when done."
        return
    fi
    log "destroying instance $CONTAINER_ID"
    local status
    status=$(curl -sS -o /tmp/destroy-response -w '%{http_code}' -X DELETE \
        "https://console.vast.ai/api/v0/instances/${CONTAINER_ID}/" \
        -H "Authorization: Bearer ${CONTAINER_API_KEY}") || status="curl-error"
    if [ "$status" = 200 ]; then
        echo "destroy accepted -- billing stops once vast.ai tears the instance down"
    else
        echo "!!! DESTROY FAILED (HTTP $status). DESTROY THIS INSTANCE IN THE CONSOLE NOW. !!!" >&2
        cat /tmp/destroy-response >&2 2>/dev/null || true
    fi
}

cleanup() {
    local code=$?
    # From here on a dropped session or Ctrl-C must not skip the destroy; the
    # only way to keep the instance is the deliberate /tmp/KEEP_INSTANCE file.
    trap '' HUP INT TERM
    trap - EXIT
    log "session ending: exit code $code after $((SECONDS / 60)) minutes"
    find "$NETS" -name '*.bin' -printf '  %s bytes  %p\n' 2>/dev/null || true

    local grace
    if [ "$code" -eq 0 ]; then
        grace=$DOWNLOAD_GRACE_MIN
        echo "SUCCESS. You have $grace minutes to download $NETS, then this instance"
        echo "destroys itself. Done early? Destroy it in the console now -- no need to wait."
    else
        grace=$FAILURE_GRACE_MIN
        echo "FAILED. Destroying in $grace minutes -- enough to read run.log, not"
        echo "enough to bill a full download window for a run that produced nothing."
    fi
    echo "To keep the instance instead: touch /tmp/KEEP_INSTANCE"
    sleep $((grace * 60))
    destroy_instance
    exit "$code"
}

arm_cleanup() {
    trap cleanup EXIT HUP INT TERM
    echo "self-destroy armed: any exit from here ends in a destroyed instance"
}

# ---------------------------------------------------------------------------
# The work.
# ---------------------------------------------------------------------------

need_rust() {
    if ! command -v cargo >/dev/null 2>&1; then
        log "installing Rust (official rustup, minimal profile)"
        with_cap bash -c "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal"
        # shellcheck disable=SC1091
        source "$HOME/.cargo/env"
    fi
    cargo --version
}

build() {
    log "building the trainer with CUDA (Cargo.lock pins every bullet dependency)"
    (cd "$TRAINER" && with_cap cargo build --release --locked --features cuda)
    log "building netcheck (the engine, to check the network on this box)"
    (cd "$ENGINE" && with_cap cargo build --release --locked --features datagen --bin netcheck)
}

newest_net_since() {
    find "$NETS" -name '*.bin' -newer "$1" -printf '%T@ %p\n' 2>/dev/null \
        | sort -n | tail -1 | cut -d' ' -f2-
}

# HIDDEN = 128: 98,689 i16 values = 197,378 bytes, padded to a multiple of 64.
check_size() {
    local size
    size=$(stat -c %s "$1")
    echo "$1 ($size bytes)"
    if [ "$size" != 197378 ] && [ "$size" != 197440 ]; then
        echo "UNEXPECTED SIZE: the engine loader will reject this. Check HIDDEN_SIZE." >&2
        exit 1
    fi
}

netcheck_gate() {
    log "netcheck: $(basename "$1") against $(basename "$2")"
    if ! "$ENGINE/target/release/netcheck" "$1" "$2" 200000; then
        echo "netcheck FAILED -- the network does not track its data." >&2
        echo "Stopping here rather than spend more on a pipeline that is wrong." >&2
        exit 4
    fi
}

train() {
    local data="$1" id="$2" epochs="$3" batch="$4" wdl="${5:-0.4}"
    local bytes positions batches stamp net
    bytes=$(stat -c %s "$data")
    [ $((bytes % 32)) -eq 0 ] || { echo "$data is not whole 32-byte records" >&2; exit 1; }
    positions=$((bytes / 32))
    # One superbatch = one pass over the data. bullet's default of ~100M
    # positions per superbatch would loop a smaller file dozens of times each.
    batches=$(( (positions + batch - 1) / batch ))
    stamp=$(mktemp)
    log "training $id on $positions positions: $epochs superbatches of $batches x $batch, wdl $wdl"
    with_cap "$TRAINER/target/release/newchessbot-trainer" --data "$data" --out "$NETS" \
        --id "$id" --wdl "$wdl" --batch-size "$batch" --batches-per-superbatch "$batches" \
        --superbatches "$epochs" --lr-step $((epochs * 45 / 100 > 0 ? epochs * 45 / 100 : 1))
    net=$(newest_net_since "$stamp")
    [ -n "$net" ] || { echo "training finished but wrote no .bin network" >&2; exit 1; }
    check_size "$net"
    TRAINED_NET="$net"
}

need_disk() {
    local need_gb="$1" free_gb
    free_gb=$(df -P -BG "$BUNDLE" | awk 'NR == 2 { gsub("G", "", $4); print $4 }')
    log "checking disk: $free_gb GB free, $need_gb GB needed"
    if [ "$free_gb" -lt "$need_gb" ]; then
        echo "Not enough disk for this run. Destroy the instance and rent one with more." >&2
        exit 1
    fi
}

# Stream the Lichess evaluation database into the converter, then turn the
# result into shuffled bulletformat. Each intermediate is deleted once the next
# stage has been verified, and none of the 22 GB download is ever stored.
prepare_lichess() {
    local text="$DATA/lichess.txt" raw="$DATA/lichess.data" shuffled="$DATA/lichess.shuffled.data"

    log "building bullet-utils at the trainer's bullet revision"
    with_cap cargo install --locked --git https://github.com/jw1912/bullet --rev "$BULLET_REV" \
        bullet-utils --root "$BUNDLE/tools"
    local utils="$BUNDLE/tools/bin/bullet-utils"

    log "building lichesseval"
    (cd "$ENGINE" && with_cap cargo build --release --locked --features datagen --bin lichesseval)

    log "streaming the Lichess evaluation database into lichesseval"
    with_cap bash -c '
        curl -sSfL --retry 3 "$1" | "$2" - "$3" $4
        status=("${PIPESTATUS[@]}")
        [ "${status[1]}" -eq 0 ] || exit "${status[1]}"
        # With a MAX_POSITIONS limit the reader stops early and curl dies of
        # SIGPIPE, which is expected. Without one, a curl failure is real.
        [ -n "$4" ] || [ "${status[0]}" -eq 0 ] || exit "${status[0]}"
    ' _ "$LICHESS_URL" "$ENGINE/target/release/lichesseval" "$text" "${MAX_POSITIONS:-}" \
        2>&1 | tee "$DATA/lichesseval.log"
    if [ -z "${MAX_POSITIONS:-}" ] && grep -q "mid-frame" "$DATA/lichesseval.log"; then
        echo "WARNING: the download ended early; training on the positions that arrived." >&2
    fi

    local lines records errors
    lines=$(wc -l < "$text")
    # A sample for netcheck, taken before the text is deleted. It overlaps the
    # training set, so it tests the pipeline, not generalisation.
    shuf -n 200000 "$text" > "$DATA/lichess.check.txt"

    log "converting $lines positions to bulletformat"
    with_cap "$utils" convert --from text --input "$text" --output "$raw" --threads "$(nproc)" \
        > "$DATA/convert.log"
    tail -2 "$DATA/convert.log"
    errors=$(grep -ci "error parsing" "$DATA/convert.log" || true)
    records=$(( $(stat -c %s "$raw") / 32 ))
    if [ "$errors" != 0 ] || [ "$records" != "$lines" ]; then
        echo "conversion lost positions: $errors parse errors, $records records from $lines lines" >&2
        exit 1
    fi
    rm -f "$text"

    log "validating"
    "$utils" validate --input "$raw" > "$DATA/validate.log"
    tail -3 "$DATA/validate.log"
    grep -q "No invalid positions!" "$DATA/validate.log" \
        || { echo "bullet validate found invalid positions" >&2; exit 1; }

    log "shuffling"
    with_cap "$utils" shuffle --input "$raw" --output "$shuffled" --mem-used-mb "${SHUFFLE_MB:-8192}"
    [ "$(stat -c %s "$raw")" = "$(stat -c %s "$shuffled")" ] \
        || { echo "shuffle changed the file size" >&2; exit 1; }
    rm -f "$raw"

    TRAIN_DATA="$shuffled"
    TRAIN_TEXT="$DATA/lichess.check.txt"
}

# `selfplay` trains on the datagen output in the bundle; `lichess` fetches
# and converts the Lichess database on this box first.
session() {
    local mode="$1" required
    if [ "$mode" = lichess ]; then
        required="$SMOKE_DATA $SMOKE_TEXT"
    else
        required="$SMOKE_DATA $SMOKE_TEXT $TRAIN_DATA $TRAIN_TEXT"
    fi
    for f in $required; do
        [ -f "$f" ] || { echo "missing $f -- was this bundle made by prepare.sh?" >&2; exit 1; }
    done

    need_cuda
    need_self_destroy
    need_budget
    if [ "$mode" = lichess ]; then
        need_disk "$MIN_FREE_GB_LICHESS"
    fi
    arm_cleanup

    need_rust
    build

    # Enough epochs over 16K positions to learn them properly, so netcheck tests
    # the loader and perspective rather than merely catching an undertrained net.
    # Runs before the Lichess download, so a broken build costs minutes, not
    # the whole stream.
    train "$SMOKE_DATA" smoke 20 1024
    netcheck_gate "$TRAINED_NET" "$SMOKE_TEXT"

    local default_wdls="0.0 0.4"
    if [ "$mode" = lichess ]; then
        prepare_lichess
        # No game results in that data, so only the score can be learned.
        default_wdls="0.0"
    fi

    # One net per WDL weight. Training is seconds per net at this size, so a
    # sweep costs almost nothing beyond the setup already paid for. On
    # self-play data WDL 0.0 (search score only) is the control: it should all
    # but reproduce the evaluation that labelled the data.
    local wdl finals=""
    for wdl in ${WDLS:-$default_wdls}; do
        train "$TRAIN_DATA" "newchessbot-$mode-wdl$wdl" "$EPOCHS" 16384 "$wdl"
        netcheck_gate "$TRAINED_NET" "$TRAIN_TEXT"
        finals="$finals $TRAINED_NET"
    done

    log "done"
    echo "final networks:"
    for net in $finals; do echo "  $net"; done
}

# Sourcing defines the functions without running anything, so the guardrails
# can be tested locally without a GPU or a rental.
[ "${BASH_SOURCE[0]}" = "$0" ] || return 0

case "${1:-}" in
    all) session selfplay ;;
    lichess) session lichess ;;
    check)
        need_cuda
        need_self_destroy
        need_budget
        echo
        echo "checks passed. Run: nohup bash trainer/vast/run.sh all > run.log 2>&1 &"
        ;;
    *)
        echo "usage: bash trainer/vast/run.sh all       # guarded session on the bundled self-play data"
        echo "       bash trainer/vast/run.sh lichess   # guarded session on the Lichess evaluation database"
        echo "       bash trainer/vast/run.sh check     # fast checks only"
        exit 2
        ;;
esac
