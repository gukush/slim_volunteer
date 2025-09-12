#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="../kernels/cpp"
SRC="${OUT_DIR}/block_matmul.cpp"
OUT="${OUT_DIR}/block_matmul.js"

echo "Building ${SRC} -> ${OUT}"
emcc "${SRC}" -O3 \
  -s MODULARIZE=1 \
  -s EXPORT_ES6=1 \
  -s ENVIRONMENT=web \
  -s SINGLE_FILE=1 \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s EXPORTED_FUNCTIONS='["_matmul","_malloc","_free"]' \
  -s EXPORTED_RUNTIME_METHODS='["cwrap","HEAPU8","HEAP8","HEAP16","HEAPU16","HEAP32","HEAPU32","HEAPF32","wasmMemory"]' \
  -o "${OUT}"

echo "Done. Generated ${OUT} (self-contained)."
