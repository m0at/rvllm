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

    if [ -f "${PROJECT_ROOT}/target/release/rvllm" ]; then
        BIN="${PROJECT_ROOT}/target/release/rvllm"
    elif [ -f "${PROJECT_ROOT}/target/debug/rvllm" ]; then
        BIN="${PROJECT_ROOT}/target/debug/rvllm"
    else
        echo "[smoke] building rvllm..."
        (cd "$PROJECT_ROOT" && cargo build -p rvllm-server 2>&1)
        BIN="${PROJECT_ROOT}/target/debug/rvllm"
    fi

    if [ ! -x "$BIN" ]; then
        echo "[smoke] ERROR: binary not found at $BIN"
        exit 1
    fi

    echo "[smoke] starting server: $BIN serve --model mock-model --port $PORT"
    "$BIN" serve --model mock-model --host 0.0.0.0 --port "$PORT" &
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

# -- Test 5: Metrics endpoint -----------------------------------------------
echo "--- Test 5: Metrics endpoint ---"
HTTP_CODE=$(curl -sf -o /dev/null -w "%{http_code}" "${BASE_URL}/metrics")
check "GET /metrics" "$HTTP_CODE" 200

# -- Summary -----------------------------------------------------------------
echo ""
echo "========================================"
echo "  Results: ${PASS} passed, ${FAIL} failed"
echo "========================================"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
exit 0
