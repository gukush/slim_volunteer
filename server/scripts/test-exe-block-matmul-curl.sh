#!/bin/bash
# Simple test script for exe-block-matmul-flex strategy using curl

set -e

# Default parameters
N=${N:-16}
K=${K:-16}
M=${M:-16}
BACKEND=${BACKEND:-opencl}
HOST=${HOST:-https://localhost:3000}

echo "🧪 Testing exe-block-matmul-flex strategy with ${BACKEND} backend"
echo "📊 Matrix dimensions: ${N}x${K} * ${K}x${M} = ${N}x${M}"

# Generate test matrices
echo "📝 Generating test matrices..."
python3 -c "
import numpy as np
import sys

N, K, M = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])

# Generate random matrices
A = np.random.randn(N, K).astype(np.float32)
B = np.random.randn(K, M).astype(np.float32)

# Save as binary files
A.tofile('/tmp/A_test.bin')
B.tofile('/tmp/B_test.bin')

print(f'✓ Generated A.bin ({A.size} elements) and B.bin ({B.size} elements)')

# Compute reference result
C_ref = np.dot(A, B)
C_ref.tofile('/tmp/C_ref.bin')
print(f'✓ Computed reference result ({C_ref.size} elements)')
" $N $K $M

# Create task
echo "📤 Creating task..."
TASK_RESPONSE=$(curl -k -s -X POST \
  -F "strategyId=native-block-matmul-flex" \
  -F "K=1" \
  -F "label=exe-bm-flex-test" \
  -F "config={\"N\":$N,\"K\":$K,\"M\":$M,\"backend\":\"$BACKEND\"}" \
  -F "A.bin=@/tmp/A_test.bin" \
  -F "B.bin=@/tmp/B_test.bin" \
  "$HOST/tasks")

if echo "$TASK_RESPONSE" | grep -q "error"; then
  echo "❌ Create task failed: $TASK_RESPONSE"
  exit 1
fi

TASK_ID=$(echo "$TASK_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['id'])")
echo "✓ Created task $TASK_ID"

# Start task
echo "🚀 Starting task..."
START_RESPONSE=$(curl -k -s -X POST "$HOST/tasks/$TASK_ID/start")
if echo "$START_RESPONSE" | grep -q "error"; then
  echo "❌ Start failed: $START_RESPONSE"
  exit 1
fi
echo "✓ Task started"

# Monitor task progress
echo "⏳ Monitoring task progress..."
while true; do
  STATUS_RESPONSE=$(curl -k -s "$HOST/tasks/$TASK_ID")
  STATUS=$(echo "$STATUS_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['status'])")
  COMPLETED=$(echo "$STATUS_RESPONSE" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('completedChunks', 0))")
  TOTAL=$(echo "$STATUS_RESPONSE" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('totalChunks', '?'))")

  echo -n "⏱️  status=$STATUS $COMPLETED/$TOTAL   \r"

  if [ "$STATUS" = "completed" ] || [ "$STATUS" = "error" ] || [ "$STATUS" = "canceled" ]; then
    break
  fi
  sleep 1
done
echo

if [ "$STATUS" != "completed" ]; then
  echo "❌ Task did not complete: $STATUS"
  exit 2
fi
echo "✅ Task completed successfully"

# Download and validate results
echo "📥 Downloading results..."
curl -k -s -o /tmp/C_result.bin "$HOST/tasks/$TASK_ID/output"

echo "🔍 Validating results..."
python3 -c "
import numpy as np
import sys

# Load results
C_ref = np.fromfile('/tmp/C_ref.bin', dtype=np.float32)
C_result = np.fromfile('/tmp/C_result.bin', dtype=np.float32)

print(f'Reference shape: {C_ref.shape}')
print(f'Result shape: {C_result.shape}')

if C_ref.shape != C_result.shape:
    print('❌ Shape mismatch!')
    sys.exit(1)

# Compute errors
abs_error = np.abs(C_result - C_ref)
rel_error = abs_error / (np.abs(C_ref) + 1e-8)

max_abs_error = np.max(abs_error)
max_rel_error = np.max(rel_error)

print(f'📊 Validation results:')
print(f'   max absolute error: {max_abs_error:.2e}')
print(f'   max relative error: {max_rel_error:.2e}')

# Check if results are close enough
abs_tol = 1e-5
rel_tol = 1e-3

if max_abs_error < abs_tol or max_rel_error < rel_tol:
    print('✅ PASS - Results match within tolerance')
    print('🎉 exe-block-matmul-flex strategy works correctly!')
    sys.exit(0)
else:
    print('❌ FAIL - Results do not match within tolerance')
    print(f'   Worst absolute error: {max_abs_error:.2e}')
    print(f'   Worst relative error: {max_rel_error:.2e}')
    sys.exit(1)
"

echo "🧹 Cleaning up..."
rm -f /tmp/A_test.bin /tmp/B_test.bin /tmp/C_ref.bin /tmp/C_result.bin
