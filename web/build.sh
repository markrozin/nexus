#!/usr/bin/env bash
# Build the browser page: compile the engine to WebAssembly and inline it.
#
#   bash web/build.sh        # writes dist/web/nexus.html
#
# The module is embedded as base64 rather than fetched at run time, so the page
# is one self-contained file that works wherever it is hosted.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WASM="$ROOT/web/target/wasm32-unknown-unknown/release/nexus_web.wasm"
PAGE="$ROOT/web/page.html"
OUT="$ROOT/dist/web/nexus.html"

command -v rustup > /dev/null || { echo "rustup is required" >&2; exit 1; }
rustup target list --installed | grep -qx wasm32-unknown-unknown \
    || { echo "run: rustup target add wasm32-unknown-unknown" >&2; exit 1; }

echo "building the engine for wasm32-unknown-unknown"
(cd "$ROOT/web" && cargo build --release --target wasm32-unknown-unknown)
[ -f "$WASM" ] || { echo "no wasm at $WASM" >&2; exit 1; }

mkdir -p "$(dirname "$OUT")"
python - "$PAGE" "$WASM" "$OUT" <<'PY'
import base64
import sys

page_path, wasm_path, out_path = sys.argv[1:4]
page = open(page_path, encoding="utf-8").read()
wasm = open(wasm_path, "rb").read()
marker = "__WASM_BASE64__"
if marker not in page:
    sys.exit("the page template no longer has the %s placeholder" % marker)
open(out_path, "w", encoding="utf-8", newline="\n").write(
    page.replace(marker, base64.b64encode(wasm).decode("ascii"))
)
print(f"{len(wasm):,} bytes of wasm -> {out_path}")
PY

printf 'page: %s bytes\n' "$(stat -c %s "$OUT")"
