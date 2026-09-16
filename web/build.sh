#!/usr/bin/env bash
# Build the browser page: compile the engine to WebAssembly and inline it.
#
#   bash web/build.sh
#
# Writes two files from one template:
#
#   dist/web/nexus.html           a standalone document, for hosting anywhere
#   dist/web/nexus-artifact.html  the same page without a <head>, which the
#                                 Artifact tool supplies itself
#
# The module is embedded as base64 rather than fetched at run time, so either
# file is self-contained and needs nothing alongside it.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WASM="$ROOT/web/target/wasm32-unknown-unknown/release/nexus_web.wasm"
PAGE="$ROOT/web/page.html"
OUT="$ROOT/dist/web/nexus.html"
FRAGMENT="$ROOT/dist/web/nexus-artifact.html"

command -v rustup > /dev/null || { echo "rustup is required" >&2; exit 1; }
rustup target list --installed | grep -qx wasm32-unknown-unknown \
    || { echo "run: rustup target add wasm32-unknown-unknown" >&2; exit 1; }

echo "building the engine for wasm32-unknown-unknown"
(cd "$ROOT/web" && cargo build --release --target wasm32-unknown-unknown)
[ -f "$WASM" ] || { echo "no wasm at $WASM" >&2; exit 1; }

mkdir -p "$(dirname "$OUT")"
python - "$PAGE" "$WASM" "$OUT" "$FRAGMENT" <<'PY'
import base64
import sys

page_path, wasm_path, out_path, fragment_path = sys.argv[1:5]
page = open(page_path, encoding="utf-8").read()
wasm = open(wasm_path, "rb").read()

marker = "__WASM_BASE64__"
if marker not in page:
    sys.exit("the page template no longer has the %s placeholder" % marker)
body = page.replace(marker, base64.b64encode(wasm).decode("ascii"))

# The Artifact tool supplies the document shell, charset included. Anywhere
# else the page declares its own: a server that omits the header leaves the
# browser guessing, and the chess glyphs come out as mojibake.
head = (
    '<!doctype html>\n<html lang="en">\n<head>\n'
    '<meta charset="utf-8">\n'
    '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
    "</head>\n"
    '<body style="margin: 0">\n'
)

with open(fragment_path, "w", encoding="utf-8", newline="\n") as out:
    out.write(body)
with open(out_path, "w", encoding="utf-8", newline="\n") as out:
    out.write(head + body + "\n</body>\n</html>\n")

print(f"{len(wasm):,} bytes of wasm inlined")
PY

printf 'standalone %s bytes, artifact fragment %s bytes\n' \
    "$(stat -c %s "$OUT")" "$(stat -c %s "$FRAGMENT")"
