#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Smoke test for the Co-founder OS Gateway.
#
# All values come from environment variables. No credentials are hard-coded.
# Required environment variables (with safe defaults for local testing):
#   GATEWAY_URL      – base URL of the running gateway (default: http://127.0.0.1:9000)
#   GATEWAY_AUDIT_TOKEN – bearer token for /audit/recent (default: test-token)
#   OPENAI_API_KEY   – required for OpenAI requests
#   QWEN_API_KEY     – required for Qwen requests
#   STEP_API_KEY     – required for Step requests
# ---------------------------------------------------------------------------

set -euo pipefail

GATEWAY_URL="${GATEWAY_URL:-http://127.0.0.1:9000}"
AUDIT_TOKEN="${GATEWAY_AUDIT_TOKEN:-test-token}"
OPENAI_KEY="${OPENAI_API_KEY:-}"
QWEN_KEY="${QWEN_API_KEY:-}"
STEP_KEY="${STEP_API_KEY:-}"

pass=0
fail=0

check() {
    local label="$1"
    local expected_code="$2"
    local method="$3"
    local url="$4"
    shift 4
    local headers=()
    local data=()
    while [[ $# -gt 0 ]]; do
        if [[ "$1" == "--header" ]]; then
            headers+=("$2")
            shift 2
        elif [[ "$1" == "--data" ]]; then
            data+=("$2")
            shift 2
        else
            echo "Unknown arg: $1" >&2
            exit 1
        fi
    done

    local tmp_body
    tmp_body=$(mktemp)
    local code
    code=$(
        curl --silent --output "$tmp_body" --write-out "%{http_code}" \
            -X "$method" \
            "${headers[@]}" \
            "${data[@]}" \
            "$url" 2>/dev/null
    ) || code="000"

    if [[ "$code" == "$expected_code" ]]; then
        echo "  PASS: $label (HTTP $code)"
        pass=$((pass + 1))
    else
        echo "  FAIL: $label — expected HTTP $expected_code, got HTTP $code" >&2
        echo "  Body: $(head -c 300 "$tmp_body")" >&2
        fail=$((fail + 1))
    fi
    rm -f "$tmp_body"
}

echo "=== Gateway Smoke Test ==="
echo "Target: $GATEWAY_URL"
echo ""

# 1. GET /health
echo "[1/6] GET /health"
check "health" 200 GET "$GATEWAY_URL/api/health"

# 2. authenticated GET /v1/models (requires any API key in Authorization)
echo "[2/6] GET /v1/models"
if [[ -n "$OPENAI_KEY" ]]; then
    check "list_models" 200 GET "$GATEWAY_URL/api/v1/models" \
        --header "Authorization: Bearer $OPENAI_KEY"
else
    echo "  SKIP: OPENAI_API_KEY not set"
fi

# 3. cofounder-qwen request
echo "[3/6] POST /v1/chat/completions (cofounder-qwen)"
if [[ -n "$QWEN_KEY" ]]; then
    check "qwen_chat" 200 POST "$GATEWAY_URL/api/v1/chat/completions" \
        --header "Content-Type: application/json" \
        --header "Authorization: Bearer $QWEN_KEY" \
        --data '{"provider":"cofounder-qwen","model":"'"${QWEN_MODEL:-qwen-turbo}"'","messages":[{"role":"user","content":"ping"}]}'
else
    echo "  SKIP: QWEN_API_KEY not set"
fi

# 4. cofounder-step request
echo "[4/6] POST /v1/chat/completions (cofounder-step)"
if [[ -n "$STEP_KEY" ]]; then
    check "step_chat" 200 POST "$GATEWAY_URL/api/v1/chat/completions" \
        --header "Content-Type: application/json" \
        --header "Authorization: Bearer $STEP_KEY" \
        --data '{"provider":"cofounder-step","model":"'"${STEP_MODEL:-step-2-16k}"'","messages":[{"role":"user","content":"ping"}]}'
else
    echo "  SKIP: STEP_API_KEY not set"
fi

# 5. cofounder-auto request (no provider specified → uses default)
echo "[5/6] POST /v1/chat/completions (auto/default)"
if [[ -n "$OPENAI_KEY" ]]; then
    check "auto_chat" 200 POST "$GATEWAY_URL/api/v1/chat/completions" \
        --header "Content-Type: application/json" \
        --header "Authorization: Bearer $OPENAI_KEY" \
        --data '{"messages":[{"role":"user","content":"ping"}]}'
else
    echo "  SKIP: OPENAI_API_KEY not set"
fi

# 6. authenticated GET /audit/recent
echo "[6/6] GET /audit/recent"
check "audit_recent" 200 GET "$GATEWAY_URL/api/audit/recent" \
    --header "X-Audit-Token: $AUDIT_TOKEN"

echo ""
echo "=== Results: $pass passed, $fail failed ==="
if [[ "$fail" -gt 0 ]]; then
    exit 1
fi
