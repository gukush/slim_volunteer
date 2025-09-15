#!/usr/bin/env bash
set -euo pipefail

cd /app

BASE_URL="https://localhost:3000"
N="0xDC0EC703D337B2D473CA7E558B833B9AD701331661430F17F90792F99DA3A50399"
GCD_MODE=1
REPEATS=10

# Configs: B1:total_curves:chunk_size
CONFIGS=(
  "500000:8192:2048"
  "1000000:8192:2048"
  "2000000:8192:2048"
  "4000000:8192:2048"
  "8000000:8192:2048"
  "1000000:12288:2048"
  "1000000:16384:2048"
  "1000000:24576:2048"
  "1000000:8192:4096"
  "1000000:16384:3072"
)

run_cfg () {
  local B1="$1"
  local CURVES="$2"
  local CHUNK="$3"
  for i in $(seq 1 "$REPEATS"); do
    echo "[$(date -Is)] ECM Stage1 | WebGPU | iter $i/$REPEATS | B1=$B1 curves=$CURVES chunk=$CHUNK"
    NODE_TLS_REJECT_UNAUTHORIZED=0 node ./scripts/test-ecm-stage1.mjs \
      --baseURL="$BASE_URL" \
      --N="$N" \
      --B1="$B1" \
      --total_curves="$CURVES" \
      --chunk_size="$CHUNK" \
      --gcdMode="$GCD_MODE"
  done
}

for cfg in "${CONFIGS[@]}"; do
  IFS=":" read -r B1 CURVES CHUNK <<< "$cfg"
  echo "==== Starting config: B1=$B1 | total_curves=$CURVES | chunk_size=$CHUNK ===="
  run_cfg "$B1" "$CURVES" "$CHUNK"
  echo "==== Finished config: B1=$B1 | total_curves=$CURVES | chunk_size=$CHUNK ===="
done

echo "All WebGPU ECM Stage 1 runs completed."
