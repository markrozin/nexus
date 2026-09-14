#!/usr/bin/env bash
# Destroy EVERY instance on the vast.ai account and verify the list is empty.
#
#   bash trainer/vast/reap.sh
#
# The last line of defence against a forgotten or orphaned instance. Destroy,
# never stop: a stopped instance still bills for its disk. Exits non-zero if
# the account cannot be confirmed empty.

set -uo pipefail

V="${VASTAI:-/c/Users/Mark/AppData/Roaming/Python/Python314/Scripts/vastai.exe}"
PY="${PYTHON:-python}"

list_ids() {
    "$V" show instances --raw 2>/dev/null \
        | "$PY" -c 'import json, sys; d = json.load(sys.stdin); rows = d if isinstance(d, list) else d.get("instances", []); print(" ".join(str(r["id"]) for r in rows))'
}

for attempt in $(seq 1 30); do
    if ! ids=$(list_ids); then
        echo "could not read the instance list (attempt $attempt)"
        sleep 20
        continue
    fi
    if [ -z "$ids" ]; then
        echo "VERIFIED: the account has no instances"
        exit 0
    fi
    echo "attempt $attempt: destroying $ids"
    for id in $ids; do
        "$V" destroy instance "$id" -y > /dev/null 2>&1
    done
    sleep 20
done
echo "!!! COULD NOT VERIFY THE ACCOUNT IS EMPTY -- CHECK THE CONSOLE NOW !!!"
exit 1
