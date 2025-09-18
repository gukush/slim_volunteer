#!/bin/bash
# Test script for Vulkan block matmul using the enhanced cached test

echo "🚀 Testing Vulkan Block Matmul with Native Executor"
echo "=================================================="

# Test with small matrix first
echo "📊 Testing small matrix (64x64x64)..."
node test-block-matmul-cached-enhanced.mjs \
  --framework=native \
  --backend=vulkan \
  --N=64 --K=64 --M=64 \
  --tileSize=16 \
  --validate=true

echo ""
echo "📊 Testing medium matrix (256x256x256)..."
node test-block-matmul-cached-enhanced.mjs \
  --framework=native \
  --backend=vulkan \
  --N=256 --K=256 --M=256 \
  --tileSize=32 \
  --validate=true

echo ""
echo "📊 Testing large matrix (512x512x512)..."
node test-block-matmul-cached-enhanced.mjs \
  --framework=native \
  --backend=vulkan \
  --N=512 --K=512 --M=512 \
  --tileSize=32 \
  --validate=true

echo ""
echo "🎉 All Vulkan tests completed!"

