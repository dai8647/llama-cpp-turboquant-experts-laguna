#!/usr/bin/env bash
# bench_dspark.sh - compare DeepSeek V4 decode speed: baseline vs MTP vs DSpark.
#
# The fork already ships the DSpark/MTP speculative engine (upstream #25784).
# ds4 (DwarfStar) uses the same DSpark checkpoint for its multi-token decode.
# This script measures the practical gain on YOUR hardware by running the same
# prompt through llama-server with and without speculative decoding.
#
# Usage:
#   MODEL=/path/to/deepseek-v4.gguf ./scripts/bench_dspark.sh
#   MODEL=... PROMPT_FILE=code.txt NGL=99 EXTRA="--cpu-moe" ./scripts/bench_dspark.sh
#
# Notes:
#   - The first run with --mtp / --dflash downloads the support GGUF from HF
#     (takes a while, then cached).
#   - Keep the same model, context, and GPU flags across cases for a fair
#     comparison. Only speculative decoding should differ.
#   - Requires: llama-server built (default ./build/bin/llama-server), curl, python3.
set -euo pipefail

MODEL="${MODEL:-}"
PROMPT_FILE="${PROMPT_FILE:-}"
CTX="${CTX:-32768}"
NGL="${NGL:-99}"
PORT="${PORT:-18080}"
SERVER="${SERVER:-./build/bin/llama-server}"
EXTRA="${EXTRA:-}"
N_GEN="${N_GEN:-128}"
TEMPERATURE="${TEMPERATURE:-0}"

if [[ -z "$MODEL" ]]; then
    echo "error: set MODEL=/path/to/model.gguf (e.g. a mixed_quant.py output)" >&2
    exit 1
fi
if [[ ! -x "$SERVER" ]]; then
    echo "error: llama-server not found at $SERVER (build first, or set SERVER=...)" >&2
    exit 1
fi
if [[ -z "$PROMPT_FILE" ]]; then
    PROMPT_FILE="$(mktemp)"
    cat > "$PROMPT_FILE" <<'EOF'
Write a complete Python function that parses a simple arithmetic expression
with +,-,*,/ and parentheses into an AST, then evaluates it. Include a few
asserts at the end of the function body.
EOF
    trap 'rm -f "$PROMPT_FILE"' EXIT
fi

declare -A RESULTS

run_case() {
    local label="$1"; shift
    local flags="$*"

    echo "== case: $label =="
    "$SERVER" -m "$MODEL" -c "$CTX" -ngl "$NGL" --port "$PORT" \
        --host 127.0.0.1 --no-webui ${EXTRA} $flags >/tmp/dspark-bench-server.log 2>&1 &
    local pid=$!
    trap 'kill $pid 2>/dev/null || true' EXIT

    # wait for readiness (up to 10 min; sidecar downloads can be slow)
    local ok=0
    for _ in $(seq 1 600); do
        if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then ok=1; break; fi
        if ! kill -0 $pid 2>/dev/null; then break; fi
        sleep 1
    done
    if [[ "$ok" != "1" ]]; then
        echo "!! server failed to become ready; tail of log:" >&2
        tail -20 /tmp/dspark-bench-server.log >&2
        kill $pid 2>/dev/null || true
        RESULTS["$label"]="FAILED"
        return
    fi

    local body
    body=$(python3 -c 'import json,sys; print(json.dumps({"prompt": open(sys.argv[1]).read(), "max_tokens": int(sys.argv[2]), "temperature": float(sys.argv[3]), "stream": False}))' \
        "$PROMPT_FILE" "$N_GEN" "$TEMPERATURE")

    local resp
    resp=$(curl -sf -X POST "http://127.0.0.1:${PORT}/v1/completions" \
        -H 'Content-Type: application/json' -d "$body" || true)

    kill $pid 2>/dev/null || true
    trap - EXIT
    wait $pid 2>/dev/null || true

    if [[ -z "$resp" ]]; then
        echo "!! no response from server" >&2
        tail -20 /tmp/dspark-bench-server.log >&2
        RESULTS["$label"]="FAILED"
        return
    fi

    local pps dps
    pps=$(python3 -c 'import json,sys; t=json.load(sys.stdin)["timings"]; print(round(t.get("prompt_per_second",0),1))' <<< "$resp")
    dps=$(python3 -c 'import json,sys; t=json.load(sys.stdin)["timings"]; print(round(t.get("predicted_per_second",0),1))' <<< "$resp")
    echo "   prefill ${pps} t/s   decode ${dps} t/s"
    RESULTS["$label"]="$dps"
}

echo "model : $MODEL"
echo "ctx   : $CTX   ngl: $NGL   gen tokens: $N_GEN"
echo "extra : ${EXTRA:-none}"
echo

run_case "baseline      "
run_case "mtp           " --mtp
run_case "dflash        " --dflash
run_case "spec=dspark   " --spec-type draft-dspark

echo
echo "== summary (decode t/s) =="
for k in "baseline      " "mtp           " "dflash        " "spec=dspark   "; do
    printf "%-14s %s\n" "$k" "${RESULTS[$k]:-n/a}"
done
echo
echo "tip: run the same sweep on a mixed_quant.py GGUF (TQ2_0 experts) for the full ds4-style setup."
