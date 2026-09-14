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

# The deduplicated Lichess evaluations on Hugging Face (CC BY 4.0, derived from
# the Lichess CC0 database): deepest evaluation per position, White-relative.
# Streaming database.lichess.org directly failed -- it serves ~170 KB/s.
HF_BASE="https://huggingface.co/datasets/mateuszgrzyb/lichess-stockfish-normalized/resolve/main"
# name:bytes, as the Hugging Face API publishes them; downloads are checked
# against these sizes.
HF_FILES="train-00000.parquet:726163976 train-00001.parquet:727635731
          train-00002.parquet:707292699 train-00003.parquet:667565315
          train-00004.parquet:566154004 train-00005.parquet:575198222
          train-00006.parquet:633174284 train-00007.parquet:684358333
          train-00008.parquet:717004099 train-00009.parquet:552290895"
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

# Download the deduplicated Lichess evaluations from Hugging Face, convert the
# ten parquet shards in parallel, and turn the result into shuffled
# bulletformat. Each intermediate is deleted once the next stage is verified.
prepare_lichess() {
    local work="$DATA/lichess" raw="$DATA/lichess.data" shuffled="$DATA/lichess.shuffled.data"
    mkdir -p "$work"

    log "building bullet-utils at the trainer's bullet revision"
    with_cap cargo install --locked --git https://github.com/jw1912/bullet --rev "$BULLET_REV" \
        bullet-utils --root "$BUNDLE/tools"
    local utils="$BUNDLE/tools/bin/bullet-utils"

    log "building lichesseval"
    (cd "$ENGINE" && with_cap cargo build --release --locked --features datagen --bin lichesseval)
    local converter="$ENGINE/target/release/lichesseval"

    log "installing pyarrow to read parquet"
    # The image's Python lives in a venv that non-interactive shells do not
    # activate, so name it directly when it is there.
    local py=/venv/main/bin/python
    [ -x "$py" ] || py=python3
    with_cap "$py" -m pip install --quiet pyarrow
    "$py" -c "import pyarrow" || { echo "pyarrow did not install" >&2; exit 1; }

    local entry name size first=""
    log "downloading the parquet shards"
    for entry in $HF_FILES; do
        name="${entry%%:*}"
        size="${entry##*:}"
        [ -n "$first" ] || first="$name"
        with_cap curl -sSfL --retry 5 --retry-delay 5 -C - -o "$work/$name" "$HF_BASE/$name"
        if [ "$(stat -c %s "$work/$name")" != "$size" ]; then
            echo "$name is $(stat -c %s "$work/$name") bytes, expected $size" >&2
            exit 1
        fi
        echo "$name  $size bytes"
    done

    # The whole reader-to-converter path on real rows, before the long run.
    log "trial: 200,000 rows of $first through parquet2tsv and lichesseval"
    "$py" "$TRAINER/vast/parquet2tsv.py" "$work/$first" | head -200000 \
        | "$converter" --tsv - "$work/trial.txt" 2> "$work/trial.log" \
        || { cat "$work/trial.log" >&2; echo "the trial conversion failed" >&2; exit 1; }
    grep -E "^kept|not quiet|r =|PASS" "$work/trial.log"
    rm -f "$work/trial.txt"

    log "converting all shards in parallel"
    local left pids="" pid failed=0
    left=$(remaining_seconds)
    [ "$left" -gt 60 ] || { echo "SPENDING CAP REACHED: not starting the conversion" >&2; exit 3; }
    for entry in $HF_FILES; do
        name="${entry%%:*}"
        timeout --signal=INT --kill-after=60 "$left" bash -c '
            set -o pipefail
            "$1" "$2" "$3" | "$4" --tsv - "$5" $6
        ' _ "$py" "$TRAINER/vast/parquet2tsv.py" "$work/$name" "$converter" "$work/$name.txt" \
            "${MAX_POSITIONS:+$((MAX_POSITIONS / 10))}" 2> "$work/$name.log" &
        pids="$pids $!"
    done
    for pid in $pids; do
        wait "$pid" || failed=1
    done
    for entry in $HF_FILES; do
        name="${entry%%:*}"
        echo "$name: $(grep -E '^kept|PASS|FAIL' "$work/$name.log" | tr -s ' ' | tr '\n' ' ')"
    done
    [ "$failed" = 0 ] || { echo "a shard failed to convert; see $work/*.log" >&2; exit 1; }
    rm -f "$work"/*.parquet

    log "converting to bulletformat, shard by shard"
    local lines records errors total=0
    : > "$raw"
    : > "$DATA/lichess.check.txt"
    for entry in $HF_FILES; do
        name="${entry%%:*}"
        lines=$(wc -l < "$work/$name.txt")
        # A slice of every shard for netcheck, taken before the text is
        # deleted. It overlaps the training set, so it tests the pipeline, not
        # generalisation.
        shuf -n 20000 "$work/$name.txt" >> "$DATA/lichess.check.txt"
        with_cap "$utils" convert --from text --input "$work/$name.txt" --output "$work/$name.data" \
            --threads "$(nproc)" > "$work/$name.convert.log"
        errors=$(grep -ci "error parsing" "$work/$name.convert.log" || true)
        records=$(( $(stat -c %s "$work/$name.data") / 32 ))
        if [ "$errors" != 0 ] || [ "$records" != "$lines" ]; then
            echo "$name lost positions: $errors parse errors, $records records from $lines lines" >&2
            exit 1
        fi
        # Whole 32-byte records, so shards concatenate into one valid file.
        cat "$work/$name.data" >> "$raw"
        rm -f "$work/$name.txt" "$work/$name.data"
        total=$((total + records))
    done
    echo "$total positions in bulletformat"

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
