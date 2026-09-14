#!/usr/bin/env bash
# Prepare a paid GPU session, and refuse to call it ready unless every check
# that can run without a GPU has passed.
#
#   bash trainer/vast/prepare.sh data/pipeline.txt   # self-play data from datagen
#   bash trainer/vast/prepare.sh --lichess           # Lichess evals, fetched on the rental
#
# GUARDRAIL: do not rent a GPU until this prints PREFLIGHT PASSED.
#
# Everything here runs locally, for free. The point is that the only things left
# to discover on a billed instance are the ones that genuinely need a GPU -- the
# CUDA build and training itself -- and never a bad data file, a sign error, a
# CRLF script, or a trainer or engine that does not build from its lockfile.
#
# Produces dist/newchessbot-train.tar.gz, laid out as run.sh expects:
#
#   newchessbot-train/trainer/   the trainer crate and the runbook
#   newchessbot-train/engine/    the engine, so netcheck and lichesseval run on the box
#   newchessbot-train/data/      smoke data, plus the training data in self-play mode
#
# --lichess bundles no training data: the 22 GB database is streamed on the
# rental, since nothing that large can be uploaded through the browser. Its
# preflight converts the first 50 MB instead (dist/lichess_eval_head.zst) and
# requires that to survive bullet-utils intact.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [ "${1:-}" = "--lichess" ]; then
    MODE=lichess
else
    MODE=selfplay
    SRC="${1:?usage: bash trainer/vast/prepare.sh data/SOURCE.txt | --lichess}"
    STEM="${SRC%.txt}"
fi
BU="${BULLET_UTILS:-tools/bullet-utils.exe}"
SMOKE_DATA=data/tp8.shuffled.data
SMOKE_TEXT=data/tp8.dedup.txt
HF_SAMPLE=dist/hf_sample.tsv
HF_PATHOLOGICAL=dist/hf_pathological.tsv
BUNDLE=dist/newchessbot-train.tar.gz
CHECK_LINES=200000

log() { printf '\n=== %s ===\n' "$*"; }
fail() {
    printf '\nPREFLIGHT FAILED: %s\nDo not rent a GPU until this passes.\n' "$*" >&2
    exit 1
}

[ -f "$BU" ] || fail "bullet-utils not found at $BU (set BULLET_UTILS)"
[ -f "$SMOKE_DATA" ] || fail "missing the smoke-test data $SMOKE_DATA"
[ -f "$SMOKE_TEXT" ] || fail "missing the smoke-test text $SMOKE_TEXT"
[ -f networks/default.bin ] || fail "missing networks/default.bin, which the engine embeds"

log "building data tools"
cargo build --release --features datagen --bin datacheck --bin datadedup --bin netcheck --bin lichesseval 2>&1 | tail -1

if [ "$MODE" = selfplay ]; then
    [ -f "$SRC" ] || fail "missing $SRC"

    log "0. the source file must be complete"
    if tasklist //FI "IMAGENAME eq datagen.exe" 2>/dev/null | grep -qi datagen.exe; then
        fail "datagen is still running, so $SRC may be incomplete"
    fi
    echo "no datagen running"

    log "1. datacheck on the raw file"
    out=$(./target/release/datacheck.exe "$SRC") || { echo "$out"; fail "datacheck rejected $SRC"; }
    echo "$out" | grep -E "lines checked|violations|PASS"

    log "2. drop positions with no non-king pieces"
    # Dead draws that teach nothing, and exactly what bullet's validator flags.
    # Datagen now stops games before reaching them; older files still contain some.
    awk -F ' [|] ' '{ split($1, f, " "); if (f[1] ~ /[pnbrqPNBRQ]/) print }' "$SRC" > "$STEM.nobare.txt"
    before=$(wc -l < "$SRC")
    after=$(wc -l < "$STEM.nobare.txt")
    echo "kept $after of $before (dropped $((before - after)) bare-king positions)"

    log "3. dedup"
    ./target/release/datadedup.exe "$STEM.nobare.txt" "$STEM.clean.txt"

    log "4. datacheck on the cleaned file"
    out=$(./target/release/datacheck.exe "$STEM.clean.txt") || { echo "$out"; fail "datacheck rejected the cleaned file"; }
    echo "$out" | grep -E "lines checked|duplicate|violations|PASS"
    echo "$out" | grep -q "duplicate boards   0 " || fail "duplicates survived dedup"

    log "5. convert to bulletformat"
    "$BU" convert --from text --input "$STEM.clean.txt" --output "$STEM.data" --threads 4 > dist/convert.log
    tail -2 dist/convert.log
    errors=$(grep -ci "error parsing" dist/convert.log || true)
    [ "$errors" = 0 ] || fail "bullet-utils could not parse $errors lines"
    lines=$(wc -l < "$STEM.clean.txt")
    bytes=$(stat -c %s "$STEM.data")
    [ $((bytes % 32)) -eq 0 ] || fail "$STEM.data is not whole 32-byte records"
    [ $((bytes / 32)) -eq "$lines" ] || fail "converted $((bytes / 32)) records from $lines lines"
    echo "$lines positions, $bytes bytes, 32 bytes each"

    log "6. bullet validate"
    v=$("$BU" validate --input "$STEM.data")
    echo "$v" | tail -4
    echo "$v" | grep -q "No invalid positions!" || fail "bullet validate found invalid positions"

    log "7. shuffle"
    # Datagen writes whole games in sequence; unshuffled batches would be dozens of
    # near-identical positions from one game.
    "$BU" shuffle --input "$STEM.data" --output "$STEM.shuffled.data" --mem-used-mb 1024 | tail -1
    [ "$(stat -c %s "$STEM.data")" = "$(stat -c %s "$STEM.shuffled.data")" ] || fail "shuffle changed the file size"
    if cmp -s "$STEM.data" "$STEM.shuffled.data"; then fail "shuffle did not reorder anything"; fi
    echo "same size, reordered"

    log "8. a random text sample for the final on-box netcheck"
    shuf -n "$CHECK_LINES" "$STEM.clean.txt" > "$STEM.check.txt"
    ./target/release/datacheck.exe "$STEM.check.txt" > /dev/null || fail "datacheck rejected the check sample"
    echo "$(wc -l < "$STEM.check.txt") lines, datacheck PASS"
else
    log "1. lichesseval --tsv on a sample of the Hugging Face dataset"
    [ -f "$HF_SAMPLE" ] || fail "missing $HF_SAMPLE -- rows of mateuszgrzyb/lichess-stockfish-normalized as TSV"
    ./target/release/lichesseval.exe --tsv "$HF_SAMPLE" dist/lichess_head.txt 2> dist/lichesseval.log \
        || { cat dist/lichesseval.log; fail "lichesseval rejected the sample"; }
    grep -E "^kept|not quiet|r =|PASS" dist/lichesseval.log
    # Syntax only: pyarrow is installed on the rental, not here.
    python -c "import ast, sys; ast.parse(open(sys.argv[1]).read())" trainer/vast/parquet2tsv.py \
        || fail "parquet2tsv.py does not parse"
    echo "parquet2tsv.py parses"

    log "1b. the rows that stalled session 5 convert in bounded time"
    [ -f "$HF_PATHOLOGICAL" ] || fail "missing $HF_PATHOLOGICAL -- the first 600,000 rows of train-00000.parquet as TSV"
    start=$(date +%s)
    timeout 120 ./target/release/lichesseval.exe --tsv "$HF_PATHOLOGICAL" dist/pathological.txt 2> dist/pathological.log \
        || { tail -5 dist/pathological.log; fail "lichesseval took over 2 minutes on the rows that stalled session 5"; }
    echo "$(wc -l < "$HF_PATHOLOGICAL") rows in $(($(date +%s) - start))s: $(grep -E '^kept' dist/pathological.log)"

    log "2. every converted line survives bullet-utils"
    "$BU" convert --from text --input dist/lichess_head.txt --output dist/lichess_head.data --threads 4 > dist/convert.log
    errors=$(grep -ci "error parsing" dist/convert.log || true)
    [ "$errors" = 0 ] || fail "bullet-utils could not parse $errors converted lines"
    lines=$(wc -l < dist/lichess_head.txt)
    records=$(( $(stat -c %s dist/lichess_head.data) / 32 ))
    [ "$records" = "$lines" ] || fail "converted $records records from $lines lines"
    v=$("$BU" validate --input dist/lichess_head.data)
    echo "$v" | grep -q "No invalid positions!" || fail "bullet validate found invalid positions"
    echo "$lines positions convert and validate cleanly"

    log "3. run.sh builds bullet-utils at the trainer's bullet revision"
    pinned=$(grep -oE 'rev = "[0-9a-f]{40}"' trainer/Cargo.toml | grep -oE '[0-9a-f]{40}')
    grep -q "BULLET_REV=\"$pinned\"" trainer/vast/run.sh \
        || fail "BULLET_REV in run.sh does not match trainer/Cargo.toml ($pinned)"
    echo "both at $pinned"
fi

log "9. runbook, trainer and engine"
for s in run.sh prepare.sh guardtest.sh; do
    [ "$(tr -cd '\r' < "trainer/vast/$s" | wc -c)" = 0 ] || fail "$s has CRLF line endings"
    bash -n "trainer/vast/$s" || fail "$s has a syntax error"
done
(cd trainer && cargo build --release --locked 2>&1 | tail -1) || fail "trainer does not build against its lockfile"
cargo build --release --locked --features datagen --bin netcheck --bin lichesseval 2>&1 | tail -1 \
    || fail "engine tools do not build against the lockfile"
echo "scripts are LF and parse; trainer and engine tools build with --locked, as the rental will"

log "10. bundle"
work=$(mktemp -d)
b="$work/newchessbot-train"
mkdir -p "$b/trainer" "$b/engine" "$b/data" dist
cp -r trainer/Cargo.toml trainer/Cargo.lock trainer/src trainer/vast "$b/trainer/"
# networks/ too: the engine embeds networks/default.bin at compile time.
cp -r Cargo.toml Cargo.lock src benches networks "$b/engine/"
cp "$SMOKE_DATA" "$SMOKE_TEXT" "$b/data/"
if [ "$MODE" = selfplay ]; then
    cp "$STEM.shuffled.data" "$STEM.check.txt" "$b/data/"
fi
tar -czf "$BUNDLE" -C "$work" newchessbot-train
rm -rf "$work"

# Re-extract and check what will actually be uploaded, not what was intended.
check=$(mktemp -d)
tar -xzf "$BUNDLE" -C "$check"
r="$check/newchessbot-train"
need="trainer/Cargo.toml trainer/Cargo.lock trainer/src/main.rs trainer/vast/run.sh trainer/vast/parquet2tsv.py
      engine/Cargo.toml engine/Cargo.lock engine/src/nnue.rs engine/src/bin/netcheck.rs
      engine/src/bin/lichesseval.rs engine/networks/default.bin engine/benches/engine.rs
      data/$(basename "$SMOKE_DATA") data/$(basename "$SMOKE_TEXT")"
if [ "$MODE" = selfplay ]; then
    need="$need data/$(basename "$STEM.shuffled.data") data/$(basename "$STEM.check.txt")"
fi
for f in $need; do
    [ -f "$r/$f" ] || fail "bundle is missing $f"
done
[ "$(tr -cd '\r' < "$r/trainer/vast/run.sh" | wc -c)" = 0 ] || fail "run.sh in the bundle has CRLF"
bash -n "$r/trainer/vast/run.sh" || fail "run.sh in the bundle does not parse"
rm -rf "$check"

size=$(stat -c %s "$BUNDLE")
sum=$(sha256sum "$BUNDLE" | cut -d' ' -f1)

printf '\nPREFLIGHT PASSED -- safe to rent.\n'
printf '  mode      %s\n' "$MODE"
printf '  bundle    %s (%s bytes)\n' "$BUNDLE" "$size"
printf '  sha256    %s\n' "$sum"
if [ "$MODE" = selfplay ]; then
    printf '  training  %s positions\n' "$lines"
else
    printf '  training  streamed on the rental: run.sh lichess, >= %s GB disk\n' 60
fi
