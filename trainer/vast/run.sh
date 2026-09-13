#!/usr/bin/env bash
# Train the newchessbot NNUE on a rented GPU (vast.ai or similar).
#
# Upload the bundle to the instance, then from the bundle root:
#
#   bash vast/run.sh smoke                        # build, 30-second train, check the output
#   bash vast/run.sh train data/X.data [EPOCHS]   # the real run
#
# The budget rule this script is written around: paid time should be spent
# computing, never debugging. Everything that could be verified without a GPU
# already was -- the data pipeline, the sign convention, the engine-side loader,
# and that the trainer compiles against the pinned bullet API. What is left to
# discover here is only whether the CUDA build and runtime work, and `smoke`
# answers that in about a minute for a few cents.
#
# Image: an NVIDIA CUDA *devel* image, for example
#   nvidia/cuda:12.4.1-devel-ubuntu22.04
# bullet links cudart, nvrtc and cublas from $CUDA_PATH/lib64 and libcuda from
# the driver. It compiles its GPU kernels at runtime through NVRTC, so nvcc is
# not needed -- but the toolkit libraries are, and runtime-only images can lack
# them. `need_cuda` checks for exactly those before anything slow happens.
#
# When finished, DESTROY the instance. A stopped instance can still bill for its
# disk; destroying it is what stops the charges.

set -euo pipefail

# The bundle root is the trainer crate: Cargo.toml, Cargo.lock, src/, vast/, data/.
cd "$(dirname "$0")/.."

log() { printf '\n=== %s ===\n' "$*"; }

need_cuda() {
    log "checking GPU and CUDA toolkit"
    export CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
    if [ ! -d "$CUDA_PATH/lib64" ]; then
        echo "CUDA_PATH=$CUDA_PATH has no lib64: this image lacks the CUDA toolkit." >&2
        echo "Destroy it and pick an nvidia/cuda *-devel* image." >&2
        exit 1
    fi
    for lib in libcudart libnvrtc libcublas; do
        if ! ls "$CUDA_PATH"/lib64/"$lib".so* >/dev/null 2>&1; then
            echo "missing $lib in $CUDA_PATH/lib64 -- bullet will not link." >&2
            exit 1
        fi
    done
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
}

need_rust() {
    if ! command -v cargo >/dev/null 2>&1; then
        log "installing Rust (official rustup, minimal profile)"
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
        # shellcheck disable=SC1091
        source "$HOME/.cargo/env"
    fi
    cargo --version
}

build() {
    log "building trainer with CUDA (Cargo.lock pins every bullet dependency)"
    cargo build --release --locked --features cuda
}

# Newest .bin network written after the given stamp file.
newest_net_since() {
    find nets -name '*.bin' -newer "$1" -printf '%T@ %p\n' 2>/dev/null \
        | sort -n | tail -1 | cut -d' ' -f2-
}

# HIDDEN = 128: 98,689 i16 values = 197,378 bytes, padded to a multiple of 64.
EXPECTED_EXACT=197378
EXPECTED_PADDED=197440

check_size() {
    local net="$1" size
    size=$(stat -c %s "$net")
    echo "$net ($size bytes)"
    if [ "$size" != "$EXPECTED_EXACT" ] && [ "$size" != "$EXPECTED_PADDED" ]; then
        echo "UNEXPECTED SIZE: the engine loader will reject this. Check HIDDEN_SIZE." >&2
        exit 1
    fi
    echo "size matches the engine loader"
}

smoke() {
    need_cuda
    need_rust
    build

    local data=data/tp8.shuffled.data stamp net
    [ -f "$data" ] || { echo "missing $data in the bundle" >&2; exit 1; }
    stamp=$(mktemp)

    log "smoke training: 16K verified positions, 4 tiny superbatches"
    ./target/release/newchessbot-trainer --data "$data" --out nets --id smoke \
        --batch-size 1024 --batches-per-superbatch 16 --superbatches 4 \
        --lr-step 3 --save-rate 2 --threads 4

    net=$(newest_net_since "$stamp")
    [ -n "$net" ] || { echo "training finished but wrote no .bin network" >&2; exit 1; }
    log "network written"
    check_size "$net"
    echo
    echo "SMOKE PASSED in ${SECONDS}s. Download $net and run netcheck locally"
    echo "before starting the real run."
}

train() {
    local data="${1:?usage: bash vast/run.sh train data/X.data [EPOCHS]}"
    local epochs="${2:-40}"
    [ -f "$data" ] || { echo "missing $data" >&2; exit 1; }

    need_cuda
    need_rust
    build

    # bulletformat records are 32 bytes each.
    local bytes positions batch=16384 batches stamp net
    bytes=$(stat -c %s "$data")
    if [ $((bytes % 32)) -ne 0 ]; then
        echo "$data is not a whole number of 32-byte records -- corrupt or wrong format" >&2
        exit 1
    fi
    positions=$((bytes / 32))

    # One superbatch = one pass over the data. bullet's default is 6104 batches,
    # about 100M positions, which on a smaller file would loop it dozens of
    # times per superbatch and make the schedule meaningless.
    batches=$(( (positions + batch - 1) / batch ))
    stamp=$(mktemp)

    log "training on $positions positions: $epochs superbatches of $batches batches"
    ./target/release/newchessbot-trainer --data "$data" --out nets --id newchessbot \
        --batch-size "$batch" --batches-per-superbatch "$batches" --superbatches "$epochs"

    net=$(newest_net_since "$stamp")
    [ -n "$net" ] || { echo "training finished but wrote no .bin network" >&2; exit 1; }
    log "final network"
    check_size "$net"
    echo
    echo "TRAINING DONE in ${SECONDS}s. Download nets/, then destroy the instance."
}

case "${1:-}" in
    smoke) smoke ;;
    train) shift; train "$@" ;;
    *)
        echo "usage: bash vast/run.sh smoke"
        echo "       bash vast/run.sh train data/X.data [EPOCHS]"
        exit 2
        ;;
esac
