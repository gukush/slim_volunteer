#!/usr/bin/env bash
set -euo pipefail

# --- Main Script ---
cd /app

REPEATS=10
SCRIPT="/app/scripts/test-block-matmul-cached-enhanced.mjs"

# Configs: N:tileSize  (K=M=N)
CONFIGS=(
  "12288:2048"
  "10240:2048"
  "6144:2048"
  "8192:2048"
  "10240:1536"
  "10240:3072"
  "10240:4096"
  "10240:1024"
)

RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="/app/logs/matmul/webgpu/${RUN_TS}"
STUB_DIR="${LOG_DIR}/stubs" # Directory for completion stubs
SUMMARY_CSV="${LOG_DIR}/summary.csv"
mkdir -p "${LOG_DIR}"
mkdir -p "${STUB_DIR}" # Create the directory for stub files

# CSV header
if [[ ! -f "$SUMMARY_CSV" ]]; then
  echo "run_ts,backend,N,tile_size,iter,exit_code,duration_sec,started_at,finished_at,log_file" > "$SUMMARY_CSV"
fi

run_cfg () {
  local NVAL="$1"
  local TILE="$2"
  local LOG_FILE="${LOG_DIR}/N-${NVAL}_tile-${TILE}.log"

  echo "==== Starting Matmul | WebGPU | N=K=M=$NVAL | tileSize=$TILE ====" | tee -a "$LOG_FILE"

  for i in $(seq 1 "$REPEATS"); do
    # --- New: Define stub file path and check if it exists ---
    local stub_file="${STUB_DIR}/N-${NVAL}_tile-${TILE}_iter-${i}.done"
    if [[ -f "$stub_file" ]]; then
      echo "[$(date -Is)] Matmul | WebGPU | SKIP  iter $i/$REPEATS | N=$NVAL tile=$TILE (already complete)" | tee -a "$LOG_FILE"
      continue # Skip to the next iteration
    fi

    # This block now runs only if the stub file was not found
    {
      start_iso="$(date -Is)"
      start_sec="$(date +%s)"
      echo "[$start_iso] Matmul | WebGPU | BEGIN iter $i/$REPEATS | N=$NVAL tile=$TILE"

      set +e
      NODE_TLS_REJECT_UNAUTHORIZED=0 node "$SCRIPT" \
        --N="$NVAL" --K="$NVAL" --M="$NVAL" \
        --validate=false \
        --framework=webgpu \
        --tileSize="$TILE" \
        --cleanupOutput=true
      rc=$?
      set -e

      end_iso="$(date -Is)"
      end_sec="$(date +%s)"
      duration="$(( end_sec - start_sec ))"

      # --- Modified: Create stub file only on success (exit code 0) ---
      if [[ $rc -eq 0 ]]; then
        echo "[$end_iso] Matmul | WebGPU | END   iter $i/$REPEATS | exit_code=$rc | duration=${duration}s"
        touch "$stub_file" # Create the empty stub file to mark completion
      else
        echo "[$end_iso] Matmul | WebGPU | FAIL  iter $i/$REPEATS | exit_code=$rc | duration=${duration}s"
      fi
      echo

      printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
        "$RUN_TS" "webgpu" "$NVAL" "$TILE" "$i" "$rc" "$duration" "$start_iso" "$end_iso" "$LOG_FILE" >> "$SUMMARY_CSV"
    } 2>&1 | tee -a "$LOG_FILE" || true
  done

  echo "==== Finished Matmul | WebGPU | N=$NVAL | tileSize=$TILE ====" | tee -a "$LOG_FILE"
  echo
}

for cfg in "${CONFIGS[@]}"; do
  IFS=":" read -r NVAL TILE <<< "$cfg"
  run_cfg "$NVAL" "$TILE"
done

echo "All WebGPU Matmul runs completed. Logs & summary: ${LOG_DIR}"