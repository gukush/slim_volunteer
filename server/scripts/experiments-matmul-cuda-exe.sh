#!/usr/bin/env bash
set -euo pipefail

cd /app

REPEATS=10
SCRIPT="/app/scripts/test-block-matmul-cached-enhanced.mjs"

# Configs: N:tileSize  (K=M=N)
CONFIGS=(
  "10240:1024"
  "10240:1536"
  "10240:2048"
  "10240:512"
)

RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="/app/logs/matmul/cuda-exe/${RUN_TS}"
SUMMARY_CSV="${LOG_DIR}/summary.csv"
mkdir -p "${LOG_DIR}"

# CSV header
if [[ ! -f "$SUMMARY_CSV" ]]; then
  echo "run_ts,backend,N,tile_size,iter,exit_code,duration_sec,started_at,finished_at,log_file" > "$SUMMARY_CSV"
fi

run_cfg () {
  local NVAL="$1"
  local TILE="$2"
  local LOG_FILE="${LOG_DIR}/N-${NVAL}_tile-${TILE}.log"

  echo "==== Starting Matmul | CUDA exe | N=K=M=$NVAL | tileSize=$TILE ====" | tee -a "$LOG_FILE"

  for i in $(seq 1 "$REPEATS"); do
    {
      start_iso="$(date -Is)"
      start_sec="$(date +%s)"
      echo "[$start_iso] Matmul | CUDA exe | BEGIN iter $i/$REPEATS | N=$NVAL tile=$TILE"

      set +e
      NODE_TLS_REJECT_UNAUTHORIZED=0 node "$SCRIPT" \
        --N="$NVAL" --K="$NVAL" --M="$NVAL" \
        --validate=false \
        --framework=exe --backend=cuda \
        --tileSize="$TILE" \
        --cleanupOutput=true
      rc=$?
      set -e

      end_iso="$(date -Is)"
      end_sec="$(date +%s)"
      duration="$(( end_sec - start_sec ))"

      echo "[$end_iso] Matmul | CUDA exe | END   iter $i/$REPEATS | exit_code=$rc | duration=${duration}s"
      echo

      printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
        "$RUN_TS" "cuda-exe" "$NVAL" "$TILE" "$i" "$rc" "$duration" "$start_iso" "$end_iso" "$LOG_FILE" >> "$SUMMARY_CSV"
    } 2>&1 | tee -a "$LOG_FILE" || true
  done

  echo "==== Finished Matmul | CUDA exe | N=$NVAL | tileSize=$TILE ====" | tee -a "$LOG_FILE"
  echo
}

for cfg in "${CONFIGS[@]}"; do
  IFS=":" read -r NVAL TILE <<< "$cfg"
  run_cfg "$NVAL" "$TILE"
done

echo "All CUDA exe Matmul runs completed. Logs & summary: ${LOG_DIR}"
