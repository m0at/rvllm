#!/usr/bin/env bash
# smoke_test.sh -- End-to-end smoke test for rvllm server.
#
# Usage:
#   ./scripts/smoke_test.sh [HOST] [PORT]
#
# Defaults to localhost:8000. Starts the server if not already running,
# sends test requests, and validates responses.

set -euo pipefail

HOST="${1:-localhost}"
PORT="${2:-8000}"
BASE_URL="http://${HOST}:${PORT}"
PASS=0
FAIL=0
SERVER_PID=""

cleanup() {
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[smoke] stopping server (pid $SERVER_PID)"
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

check() {
    local name="$1"
    local status="$2"
    local expected="$3"
    if [ "$status" -eq "$expected" ]; then
        echo "[PASS] $name (HTTP $status)"
        PASS=$((PASS + 1))
    else
        echo "[FAIL] $name (HTTP $status, expected $expected)"
        FAIL=$((FAIL + 1))
    fi
}

# -- Check if server is already running, otherwise try to start it ----------

if ! curl -sf "${BASE_URL}/health" > /dev/null 2>&1; then
    echo "[smoke] server not reachable at ${BASE_URL}, attempting to start..."

    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
    # The active workspace is v3/; binary is `rvllm-server`.
    V3="${PROJECT_ROOT}/v3"

    if [ -f "${V3}/target/release/rvllm-server" ]; then
        BIN="${V3}/target/release/rvllm-server"
    elif [ -f "${V3}/target/debug/rvllm-server" ]; then
        BIN="${V3}/target/debug/rvllm-server"
    else
        echo "[smoke] building rvllm-server..."
        (cd "$V3" && cargo build --bin rvllm-server 2>&1)
        BIN="${V3}/target/debug/rvllm-server"
    fi

    if [ ! -x "$BIN" ]; then
        echo "[smoke] ERROR: binary not found at $BIN"
        exit 1
    fi

    # The current CLI takes `--model-dir DIR --bind ADDR:PORT`, no
    # `serve` subcommand. The smoke test needs an actual model
    # directory; if RVLLM_SMOKE_MODEL_DIR is set, use it. Otherwise
    # require an already-running server (the early reachability
    # check above) — there's no mock-model for the live server path.
    if [ -z "${RVLLM_SMOKE_MODEL_DIR:-}" ]; then
        echo "[smoke] ERROR: no server reachable at ${BASE_URL} and \
RVLLM_SMOKE_MODEL_DIR not set; can't auto-start. \
Either start rvllm-server manually or export RVLLM_SMOKE_MODEL_DIR=/path/to/model."
        exit 1
    fi
    echo "[smoke] starting server: $BIN --model-dir $RVLLM_SMOKE_MODEL_DIR --bind 0.0.0.0:$PORT"
    "$BIN" --model-dir "$RVLLM_SMOKE_MODEL_DIR" --bind "0.0.0.0:$PORT" &
    SERVER_PID=$!

    # Wait for server to be ready (up to 10 seconds)
    for i in $(seq 1 20); do
        if curl -sf "${BASE_URL}/health" > /dev/null 2>&1; then
            echo "[smoke] server ready after ~$((i / 2))s"
            break
        fi
        sleep 0.5
    done

    if ! curl -sf "${BASE_URL}/health" > /dev/null 2>&1; then
        echo "[smoke] ERROR: server failed to start within 10s"
        exit 1
    fi
fi

echo ""
echo "========================================"
echo "  rvllm Smoke Test"
echo "  Target: ${BASE_URL}"
echo "========================================"
echo ""

# -- Test 1: Health endpoint -------------------------------------------------
echo "--- Test 1: Health endpoint ---"
HTTP_CODE=$(curl -sf -o /dev/null -w "%{http_code}" "${BASE_URL}/health")
check "GET /health" "$HTTP_CODE" 200

# -- Test 2: Models endpoint -------------------------------------------------
echo "--- Test 2: Models endpoint ---"
HTTP_CODE=$(curl -sf -o /dev/null -w "%{http_code}" "${BASE_URL}/v1/models")
check "GET /v1/models" "$HTTP_CODE" 200

# -- Test 3: Completions endpoint (valid request) ---------------------------
echo "--- Test 3: Completions endpoint ---"
RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "${BASE_URL}/v1/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "mock-model",
        "prompt": "Hello, world!",
        "max_tokens": 16,
        "temperature": 0.0
    }')
HTTP_CODE=$(echo "$RESPONSE" | tail -1)
BODY=$(echo "$RESPONSE" | head -n -1)
check "POST /v1/completions" "$HTTP_CODE" 200

if echo "$BODY" | python3 -c "import sys,json; d=json.load(sys.stdin); assert len(d.get('choices',[])) > 0" 2>/dev/null; then
    echo "  -> response has choices"
    PASS=$((PASS + 1))
else
    echo "  -> WARNING: response may not have choices (could be expected with mock)"
fi

# -- Test 4: Chat completions endpoint --------------------------------------
echo "--- Test 4: Chat completions endpoint ---"
RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "${BASE_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "mock-model",
        "messages": [{"role": "user", "content": "Say hello"}],
        "max_tokens": 16,
        "temperature": 0.0
    }')
HTTP_CODE=$(echo "$RESPONSE" | tail -1)
check "POST /v1/chat/completions" "$HTTP_CODE" 200

# Test 5 was a `/metrics` HTTP probe but the current router (see
# `v3/crates/rvllm-serve/src/router.rs`) doesn't expose one. Removed
# rather than left as a guaranteed-fail check. If/when a metrics
# endpoint lands, re-add the probe alongside its router registration.

# -- Summary -----------------------------------------------------------------
echo ""
echo "========================================"
echo "  Results: ${PASS} passed, ${FAIL} failed"
echo "========================================"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
exit 0
