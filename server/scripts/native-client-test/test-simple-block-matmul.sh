#!/bin/bash
# Simple test script for native block matrix multiplication
# Tests all three frameworks: CUDA, OpenCL, and Vulkan

set -e

# Default parameters
N=${N:-32}
K=${K:-32}
M=${M:-32}
HOST=${HOST:-https://localhost:3000}

echo "🧪 Testing native block matrix multiplication"
echo "📊 Matrix dimensions: ${N}x${K} * ${K}x${M} = ${N}x${M}"

# Test each framework
frameworks=("cuda" "opencl" "vulkan")

for framework in "${frameworks[@]}"; do
    echo ""
    echo "🔧 Testing ${framework} framework..."
    echo "=========================================="

    # Run the test
    if node scripts/test-native-block-matmul.mjs \
        --framework="$framework" \
        --N="$N" \
        --K="$K" \
        --M="$M" \
        --host="$HOST"; then
        echo "✅ ${framework} test passed"
    else
        echo "❌ ${framework} test failed"
        # Continue with other frameworks even if one fails
    fi
done

echo ""
echo "🎉 All framework tests completed!"
