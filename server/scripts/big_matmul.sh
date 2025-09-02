TASK_JSON=$(curl -sk -X POST https://localhost:3000/tasks \
  -F strategyId=block-matmul-flex \
  -F label=big-matmul \
  -F K=1 \
  -F config='{"N":6000,"K":6000,"M":2000,"chunk_size":83886,"framework":"webgpu"}' \
  -F inputFiles=@./data/A.bin\;filename=A.bin \
  -F inputFiles=@./data/B.bin\;filename=B.bin)
TASK_ID=$(echo "$TASK_JSON" | jq -r '.id')
echo "TASK_ID=$TASK_ID"
echo "//////////// previous N:60000,K:60000,M:20000,chunk_size:8388608"
