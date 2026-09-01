#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-8001}"
echo "=========================================="
echo " [Gate 5 Test] Testing end-to-end inference on port $PORT"
echo "=========================================="

RES=$(curl -s -X POST "http://localhost:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"Ornith-1.5-35B-A3B-FP8","messages":[{"role":"user","content":"Respond with OK"}],"max_tokens":4}')

echo "Response: $RES"
if echo "$RES" | grep -q "choices"; then
    echo "Gate 5 PASS: Inference verified successfully."
    exit 0
else
    echo "FAIL: Unexpected inference response."
    exit 1
fi
