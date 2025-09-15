-- host-distributed-sort-optimized.lua (CUDA-based distributed sorting with batch optimization)
-- Reduces CPU-GPU round trips by processing multiple stages per kernel launch
-- Uses batch processing for significant performance improvement

print("[lua/sort-opt] *** BATCH OPTIMIZED VERSION LOADED ***")

----------------------------------------------------------------------
-- util: normalize framework tag
----------------------------------------------------------------------
local function norm_fw(fw)
  if fw == "native-opencl" then return "opencl" end
  if fw == "native-cuda"   then return "cuda"   end
  if fw == "native-vulkan" then return "vulkan" end
  return fw or "cuda"
end

----------------------------------------------------------------------
-- util: base64 (encode/decode, lenient) - only for initial data
----------------------------------------------------------------------
local B64 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'

local function b64decode(data)
  if type(data) ~= "string" or #data == 0 then return "" end
  local pad = #data % 4
  if pad > 0 then data = data .. string.rep("=", 4 - pad) end
  data = data:gsub('[^'..B64..'=]', '')
  return (data:gsub('.', function(x)
      if x == '=' then return '' end
      local r,f = '', (B64:find(x, 1, true)-1)
      for i=6,1,-1 do r = r .. (f % 2^i - f % 2^(i-1) > 0 and '1' or '0') end
      return r
    end):gsub('%d%d%d?%d?%d?%d?%d?%d?', function(x)
      if #x ~= 8 then return '' end
      local c = 0
      for i=1,8 do if x:sub(i,i) == '1' then c = c + 2^(8-i) end end
      return string.char(c)
    end))
end

local function b64encode(bytes)
  local t = {}
  for i=1,#bytes,3 do
    local a = bytes:byte(i) or 0
    local b = bytes:byte(i+1) or 0
    local c = bytes:byte(i+2) or 0
    local n = a*65536 + b*256 + c
    local i1 = math.floor(n/262144) % 64 + 1
    local i2 = math.floor(n/4096)   % 64 + 1
    local i3 = math.floor(n/64)     % 64 + 1
    local i4 = n % 64 + 1
    t[#t+1] = B64:sub(i1,i1)
    t[#t+1] = B64:sub(i2,i2)
    if (i+1) > #bytes then
      t[#t+1] = '='; t[#t+1] = '='
    elseif (i+2) > #bytes then
      t[#t+1] = B64:sub(i3,i3); t[#t+1] = '='
    else
      t[#t+1] = B64:sub(i3,i3); t[#t+1] = B64:sub(i4,i4)
    end
  end
  return table.concat(t)
end

----------------------------------------------------------------------
-- util: artifacts view + kernel pick
----------------------------------------------------------------------
local function get_artifacts_view(chunk)
  local arr = {}
  local function append_all(t)
    if type(t) ~= "table" then return end
    for _, a in ipairs(t) do arr[#arr+1] = a end
  end
  if chunk and type(chunk.artifacts) == "table" and #chunk.artifacts > 0 then
    append_all(chunk.artifacts)
  elseif type(artifacts) == "table" and #artifacts > 0 then
    append_all(artifacts)
  end
  return arr
end

local function find_artifact_source_for(fw, arts)
  fw = norm_fw(fw)
  for _, a in ipairs(arts) do
    if a.type == "text" and norm_fw(a.backend) == fw and a.bytes then
      local s = b64decode(a.bytes)
      if type(s) == "string" and #s > 0 then return s end
    end
  end
  return nil
end

----------------------------------------------------------------------
-- util: next power of 2
----------------------------------------------------------------------
local function next_pow2(x)
  if x <= 1 then return 1 end
  local p = 1
  while p < x do
    p = p * 2
  end
  return p
end

----------------------------------------------------------------------
-- util: create padded buffer with sentinel values
----------------------------------------------------------------------
local function create_padded_buffer(original_data, original_size, padded_size, ascending)
  -- Copy original data (first original_size * 4 bytes)
  local copy_bytes = math.min(#original_data, original_size * 4)
  local original_part = original_data:sub(1, copy_bytes)

  -- Create the rest with zeros
  local padding_size = (padded_size - original_size) * 4
  local padding_bytes = string.rep("\0", padding_size)
  local padded_bytes = original_part .. padding_bytes

  -- Fill padding with sentinel values
  if original_size < padded_size then
    local sentinel = ascending and 0xFFFFFFFF or 0x00000000
    local sentinel_bytes = string.char(
      sentinel % 256,
      math.floor(sentinel / 256) % 256,
      math.floor(sentinel / 65536) % 256,
      math.floor(sentinel / 16777216) % 256
    )

    for i = original_size, padded_size - 1 do
      local offset = i * 4
      padded_bytes = padded_bytes:sub(1, offset) ..
                     sentinel_bytes ..
                     padded_bytes:sub(offset + 5)
    end
  end

  return padded_bytes
end

----------------------------------------------------------------------
-- Constants
----------------------------------------------------------------------
local WORKGROUP_SIZE = 256  -- matches WebGPU workgroup size
local BATCH_SIZE = 8        -- Process up to 8 stages per batch

----------------------------------------------------------------------
-- Generate all sort stages for a given array size
----------------------------------------------------------------------
local function generate_sort_stages(padded_size)
  local stages = {}
  local k = 2
  while k <= padded_size do
    local j = math.floor(k / 2)
    while j > 0 do
      table.insert(stages, { stage = k, substage = j })
      j = math.floor(j / 2)
    end
    k = k * 2
  end
  return stages
end

----------------------------------------------------------------------
-- Optimized bitonic sort using batch processing
----------------------------------------------------------------------
local function run_optimized_bitonic_sort(stages, initial_data, padded_size, ascending, source, fw)
  local total_gpu_time = 0
  local stage_count = 0
  local current_data = initial_data

  print("[lua/sort-opt] Using batch processing approach for optimal performance")

  -- Process stages in batches to reduce CPU-GPU round trips
  for batch_start = 1, #stages, BATCH_SIZE do
    local batch_end = math.min(batch_start + BATCH_SIZE - 1, #stages)
    local batch_stages = {}

    -- Collect stages for this batch
    for i = batch_start, batch_end do
      table.insert(batch_stages, stages[i])
    end

    local num_stages = #batch_stages
    print(string.format("[lua/sort-opt] Processing batch %d-%d (%d stages)",
                       batch_start, batch_end, num_stages))

    -- Create stage data buffer
    local stage_data = ""
    for _, stage in ipairs(batch_stages) do
      local stage_bytes = string.char(
        stage.stage % 256,
        math.floor(stage.stage / 256) % 256,
        math.floor(stage.stage / 65536) % 256,
        math.floor(stage.stage / 16777216) % 256,
        stage.substage % 256,
        math.floor(stage.substage / 256) % 256,
        math.floor(stage.substage / 65536) % 256,
        math.floor(stage.substage / 16777216) % 256
      )
      stage_data = stage_data .. stage_bytes
    end

    -- Run batch kernel
    local batch_task = {
      action = "compile_and_run",
      source = source,
      entry = "execute_task_batch",
      uniforms = { padded_size, num_stages, ascending },
      inputs = {
        { b64 = b64encode(current_data) },
        { b64 = b64encode(stage_data) }
      },
      outputSizes = { padded_size * 4 },
      grid = { math.ceil(padded_size / WORKGROUP_SIZE), 1, 1 },
      block = { WORKGROUP_SIZE, 1, 1 }
    }

    local result = executor.run(fw, batch_task)
    if not result.ok then
      error("[lua/sort-opt] Batch processing failed: " .. tostring(result.error))
    end

    -- Update current data for next batch
    current_data = b64decode(result.results[1])

    -- Track timing
    local dt = tonumber(result.ms) or 0
    total_gpu_time = total_gpu_time + dt
    stage_count = stage_count + num_stages

    print(string.format("[lua/sort-opt] Batch %d-%d completed in %.1fms (%.1fms/stage)",
                       batch_start, batch_end, dt, dt / num_stages))
  end

  return current_data, total_gpu_time, stage_count
end

----------------------------------------------------------------------
-- main entry required by C++ LuaHost wrapper
----------------------------------------------------------------------
function compile_and_run(chunk)
  print("[lua/sort-opt] compile_and_run - CUDA bitonic sort (GPU buffer optimized)")
  print("[lua/sort-opt] *** DEBUG: Starting GPU buffer optimized compile_and_run ***")

  local fw = norm_fw(workload_framework or
                     (chunk.payload and chunk.payload.framework) or
                     chunk.framework or
                     (chunk.meta and chunk.meta.framework) or "cuda")

  if fw ~= "cuda" then
    print("[lua/sort-opt] WARNING: requested fw="..tostring(fw)..", but this host is tuned for CUDA. Proceeding anyway.")
  end

  local payload = chunk.payload or chunk

  -- Extract sorting parameters
  local input_b64 = assert(payload.data, "payload.data (base64) missing")
  local original_data = b64decode(input_b64)
  local original_size = assert(payload.originalSize, "payload.originalSize missing")
  local padded_size = payload.paddedSize or next_pow2(original_size)
  local ascending = payload.ascending
  if ascending == nil then ascending = true end

  print(string.format("[lua/sort-opt] Processing %d integers (padded to %d), ascending=%s",
                      original_size, padded_size, tostring(ascending)))

  -- Get kernel source
  local arts = get_artifacts_view(chunk)
  local source = find_artifact_source_for(fw, arts) or find_artifact_source_for("cuda", arts)
  if not source then
    error("No CUDA kernel source found for bitonic sort in artifacts")
  end

  -- Prepare padded data buffer
  local padded_data = create_padded_buffer(original_data, original_size, padded_size, ascending)

  -- Generate all sort stages
  local stages = generate_sort_stages(padded_size)
  print(string.format("[lua/sort-opt] Generated %d sort stages, processing in batches of %d",
                     #stages, BATCH_SIZE))

  -- Run optimized bitonic sort
  local final_data, total_gpu_time, stage_count = run_optimized_bitonic_sort(
    stages, padded_data, padded_size, ascending, source, fw
  )

  -- Extract only the original (unpadded) portion for final result
  local final_result = final_data:sub(1, original_size * 4)

  -- Create final result
  local final_result_obj = {
    ok = true,
    results = { b64encode(final_result) },
    ms = total_gpu_time,
    timings = {
      stageCount = stage_count,
      totalGpuMs = total_gpu_time,
      avgGpuMs = stage_count > 0 and (total_gpu_time / stage_count) or 0,
      workgroupSize = WORKGROUP_SIZE,
      batchSize = BATCH_SIZE,
      totalBatches = math.ceil(#stages / BATCH_SIZE),
      optimization = "batch_processing"
    }
  }

  print(string.format("[lua/sort-opt] Bitonic sort complete: %d stages in %d batches, %.1fms total GPU time",
                      stage_count, math.ceil(#stages / BATCH_SIZE), total_gpu_time))

  return final_result_obj
end

print("[lua/sort-opt] CUDA bitonic sort host (GPU buffer optimized) loaded")
