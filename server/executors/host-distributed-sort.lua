-- host-distributed-sort.lua (CUDA-based distributed sorting for native client)
-- Implements bitonic sort using CUDA kernels, matching the WebGPU executor behavior
--
-- Expected payload format:
--   { data: "base64_encoded_buffer", originalSize: N, paddedSize: P, ascending: bool }
-- The buffer contains uint32 integers to be sorted

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
-- util: base64 (encode/decode, lenient)
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
  local padded_bytes = string.rep("\0", padded_size * 4)

  -- Copy original data (first original_size * 4 bytes)
  local copy_bytes = math.min(#original_data, original_size * 4)
  padded_bytes = original_data:sub(1, copy_bytes) .. padded_bytes:sub(copy_bytes + 1)

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
-- constants
----------------------------------------------------------------------
local WORKGROUP_SIZE = 256  -- matches WebGPU workgroup size

local DEFAULT_ENTRY_BY_FW = {
  cuda = "execute_task",
  opencl = "execute_task",
  vulkan = "main"
}

local function get_entry_point(fw, payload)
  if payload and payload.entry and #payload.entry > 0 then return payload.entry end
  return DEFAULT_ENTRY_BY_FW[norm_fw(fw)] or "execute_task"
end

----------------------------------------------------------------------
-- main entry required by C++ LuaHost wrapper
----------------------------------------------------------------------
function compile_and_run(chunk)
  print("[lua/sort] compile_and_run - CUDA bitonic sort")

  local fw = norm_fw(workload_framework or
                     (chunk.payload and chunk.payload.framework) or
                     chunk.framework or
                     (chunk.meta and chunk.meta.framework) or "cuda")

  if fw ~= "cuda" then
    print("[lua/sort] WARNING: requested fw="..tostring(fw)..", but this host is tuned for CUDA. Proceeding anyway.")
  end

  local payload = chunk.payload or chunk

  -- Extract sorting parameters
  local input_b64 = assert(payload.data, "payload.data (base64) missing")
  local original_data = b64decode(input_b64)
  local original_size = assert(payload.originalSize, "payload.originalSize missing")
  local padded_size = payload.paddedSize or next_pow2(original_size)
  local ascending = payload.ascending
  if ascending == nil then ascending = true end

  print(string.format("[lua/sort] Processing %d integers (padded to %d), ascending=%s",
                      original_size, padded_size, tostring(ascending)))

  -- Get kernel source
  local arts = get_artifacts_view(chunk)
  local source = find_artifact_source_for(fw, arts) or find_artifact_source_for("cuda", arts)
  if not source then
    error("No CUDA kernel source found for bitonic sort in artifacts")
  end

  -- Prepare padded data buffer
  local padded_data = create_padded_buffer(original_data, original_size, padded_size, ascending)
  local data_size_bytes = padded_size * 4

  -- Launch configuration
  local num_groups = math.floor((padded_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
  local grid = { num_groups, 1, 1 }
  local block = { WORKGROUP_SIZE, 1, 1 }

  local entry = get_entry_point(fw, payload)

  print(string.format("[lua/sort] Grid: %dx%dx%d, Block: %dx%dx%d",
                      grid[1], grid[2], grid[3], block[1], block[2], block[3]))

  -- Execute bitonic sort stages
  local stage_count = 0
  local total_gpu_time = 0
  local current_data = padded_data

  -- Bitonic sort algorithm: for each stage k, then for each substage j
  local k = 2
  while k <= padded_size do
    local j = math.floor(k / 2)
    while j > 0 do
      -- Create task for this stage
      local task = {
        action      = "compile_and_run",
        source      = source,
        entry       = entry,
        uniforms    = { padded_size, k, j, ascending and 1 or 0 },
        inputs      = { { b64 = b64encode(current_data) } },
        outputSizes = { data_size_bytes },
        grid        = grid,
        block       = block,
      }

      local result = executor.run(fw, task)

      if type(result) ~= "table" or not result.ok then
        local err = (type(result) == "table" and result.error) or "unknown"
        error("[lua/sort] executor error in stage k=" .. k .. " j=" .. j .. ": " .. tostring(err))
      end

      -- Get result data for next iteration
      local outs = assert(result.outputs, "missing result.outputs from executor")
      local out_b64 = assert(outs[1], "missing result.outputs[1]")
      current_data = b64decode(out_b64)

      -- Track timing
      local dt = tonumber(result.ms) or 0
      total_gpu_time = total_gpu_time + dt
      stage_count = stage_count + 1

      print(string.format("[lua/sort] Stage k=%d j=%d completed in %.1fms", k, j, dt))

      j = math.floor(j / 2)
    end
    k = k * 2
  end

  -- Extract only the original (unpadded) portion for final result
  local final_result = current_data:sub(1, original_size * 4)

  -- Create final result
  local final_result_obj = {
    ok = true,
    outputs = { b64encode(final_result) },
    ms = total_gpu_time,
    timings = {
      stageCount = stage_count,
      totalGpuMs = total_gpu_time,
      avgGpuMs = stage_count > 0 and (total_gpu_time / stage_count) or 0,
      workgroupSize = WORKGROUP_SIZE
    }
  }

  print(string.format("[lua/sort] Bitonic sort complete: %d stages, %.1fms total GPU time",
                      stage_count, total_gpu_time))

  return final_result_obj
end

print("[lua/sort] CUDA bitonic sort host loaded")
