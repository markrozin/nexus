#!/usr/bin/env bash
# Local tests of the money guardrails in trainer/vast/run.sh. No GPU, no network.
RUN="$(cd "$(dirname "$0")" && pwd)/run.sh"
T=$(mktemp -d)
pass=0; failn=0
ok()  { echo "ok   - $1"; pass=$((pass+1)); }
bad() { echo "FAIL - $1"; failn=$((failn+1)); }
expect() { # name expected_code actual_code
    if [ "$2" = "$3" ]; then ok "$1 (exit $3)"; else bad "$1: expected exit $2, got $3"; fi
}

# Fake curl records its arguments and returns a chosen HTTP status.
mkdir -p "$T/bin"
cat > "$T/bin/curl" <<'EOF'
#!/usr/bin/env bash
echo "$@" >> "$FAKE_CURL_LOG"
out=""
while [ $# -gt 0 ]; do [ "$1" = -o ] && out="$2"; shift; done
[ -n "$out" ] && echo '{"success": true}' > "$out"
printf '%s' "${FAKE_STATUS:-200}"
EOF
chmod +x "$T/bin/curl"
export FAKE_CURL_LOG="$T/curl.log"

bash -n "$RUN" && ok "parses" || bad "parses"
bash "$RUN" >/dev/null 2>&1; expect "no argument prints usage" 2 $?
( unset CUDA_PATH; export CUDA_PATH="$T/nocuda"; bash "$RUN" check >/dev/null 2>&1 ); expect "check fails fast with no CUDA toolkit" 1 $?

( source "$RUN"; unset CONTAINER_ID CONTAINER_API_KEY; need_self_destroy ) >/dev/null 2>&1
expect "refuses to run without self-destroy credentials" 1 $?
( source "$RUN"; NO_AUTODESTROY=1; need_self_destroy ) >/dev/null 2>&1
expect "NO_AUTODESTROY=1 is an explicit opt-out" 0 $?

( source "$RUN"; unset PRICE_PER_HOUR; need_budget ) >/dev/null 2>&1
expect "refuses to run without PRICE_PER_HOUR" 1 $?
( source "$RUN"; PRICE_PER_HOUR=abc; need_budget ) >/dev/null 2>&1
expect "rejects a non-numeric price" 1 $?
( source "$RUN"; PRICE_PER_HOUR=0.30; MAX_DOLLARS=0.10; need_budget ) >/dev/null 2>&1
expect "rejects a cap too small to do anything" 1 $?
out=$( source "$RUN"; PRICE_PER_HOUR=0.30; MAX_DOLLARS=3; need_budget; echo "CAP=$CAP_SECONDS" )
echo "$out" | grep -q "CAP=36000" && ok "\$3 at \$0.30/hr caps at 10 hours" || bad "cap arithmetic: $out"

( source "$RUN"; CAP_SECONDS=100; with_cap echo should-not-run ) > "$T/o" 2>&1
code=$?; expect "with_cap refuses to start work the budget cannot pay for" 3 $code
grep -qx should-not-run "$T/o" && bad "with_cap ran the command anyway" || ok "with_cap did not run it"
( source "$RUN"; SECONDS=0; DOWNLOAD_GRACE_MIN=0; CAP_SECONDS=62; with_cap sleep 150 ) > "$T/o" 2>&1
code=$?; expect "with_cap kills work that outruns the cap" 124 $code
grep -q "SPENDING CAP REACHED" "$T/o" && ok "cap message printed" || bad "no cap message"
( source "$RUN"; CAP_SECONDS=99999; with_cap false ) >/dev/null 2>&1
expect "with_cap propagates a failing command" 1 $?
( source "$RUN"; CAP_SECONDS=99999; with_cap true ) >/dev/null 2>&1
expect "with_cap passes a succeeding command" 0 $?

# Destroy: the request itself.
: > "$FAKE_CURL_LOG"
out=$( PATH="$T/bin:$PATH"; source "$RUN"; CONTAINER_ID=4242; CONTAINER_API_KEY=k; destroy_instance 2>&1 )
grep -q "DELETE https://console.vast.ai/api/v0/instances/4242/" "$FAKE_CURL_LOG" \
    && ok "destroy sends DELETE for this instance" || bad "destroy request: $(cat "$FAKE_CURL_LOG")"
grep -q "Authorization: Bearer k" "$FAKE_CURL_LOG" && ok "destroy uses the injected key" || bad "no bearer key"
echo "$out" | grep -q "destroy accepted" && ok "a 200 reads as accepted" || bad "200 handling: $out"
out=$( PATH="$T/bin:$PATH"; FAKE_STATUS=401; export FAKE_STATUS; source "$RUN"; CONTAINER_ID=4242; CONTAINER_API_KEY=k; destroy_instance 2>&1 )
echo "$out" | grep -q "DESTROY FAILED" && ok "a failed destroy is shouted, not swallowed" || bad "401 handling: $out"

# KEEP_INSTANCE opt-out, using a private path so a real /tmp file is never touched.
: > "$FAKE_CURL_LOG"
out=$( PATH="$T/bin:$PATH"; source "$RUN"; CONTAINER_ID=1; CONTAINER_API_KEY=k
       eval "$(declare -f destroy_instance | sed "s#/tmp/KEEP_INSTANCE#$T/KEEP#g")"
       touch "$T/KEEP"; destroy_instance 2>&1 )
[ -s "$FAKE_CURL_LOG" ] && bad "destroyed despite KEEP_INSTANCE" || ok "KEEP_INSTANCE prevents the destroy"
rm -f "$T/KEEP"

# The whole trap: a failure after arming ends in a destroy.
: > "$FAKE_CURL_LOG"
( PATH="$T/bin:$PATH"; source "$RUN"; CONTAINER_ID=77; CONTAINER_API_KEY=k
  FAILURE_GRACE_MIN=0; NETS="$T/nets"; arm_cleanup; false; echo unreachable ) > "$T/o" 2>&1
code=$?
grep -q "instances/77/" "$FAKE_CURL_LOG" && ok "a failure after arming destroys the instance" || bad "failure did not destroy: $(cat "$T/o")"
[ "$code" != 0 ] && ok "and the failure exit code survives (exit $code)" || bad "failure exited 0"

: > "$FAKE_CURL_LOG"
( PATH="$T/bin:$PATH"; source "$RUN"; CONTAINER_ID=78; CONTAINER_API_KEY=k
  DOWNLOAD_GRACE_MIN=0; NETS="$T/nets"; arm_cleanup; exit 0 ) > "$T/o" 2>&1
grep -q "instances/78/" "$FAKE_CURL_LOG" && ok "success also destroys the instance" || bad "success did not destroy"
grep -q "SUCCESS" "$T/o" && ok "success announces the download window" || bad "no success message"

# Signal delivery (a kill or dropped session) is not tested here: Git Bash on
# Windows does not deliver signals to a subshell the way Linux does, and the
# test hung rather than proving anything. The hard cap and the console destroy
# remain the backstops for that case.

rm -rf "$T"
echo
echo "$pass passed, $failn failed"
[ "$failn" = 0 ]
