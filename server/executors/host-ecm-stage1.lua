-- host-ecm-stage1.lua  (resumable ECM Stage 1 for native CUDA)
-- Mirrors browser executor behavior:
--   * Updates header[10]=pp_start, header[11]=pp_len per pass
--   * Runs with blockDim.x = 64, gridDim.x = ceil(n/64)
--   * Reads/writes the full IO buffer (header + consts + primes + outputs + state)
--
-- Expects the workload to carry artifacts:
--   - {type:"lua", name:"host.lua"}         -> this file
--   - {type:"text", backend:"cuda", bytes}  -> CUDA kernel source
-- Optionally OpenCL as fallback (backend:"opencl") if CUDA absent.

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
  local t, pad = {}, 0
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
-- util: endian helpers to patch header words (u32 little-endian)
----------------------------------------------------------------------
local function put_u32_le(v)
  local b0 = string.char(v % 256)
  local b1 = string.char(math.floor(v / 256) % 256)
  local b2 = string.char(math.floor(v / 65536) % 256)
  local b3 = string.char(math.floor(v / 16777216) % 256)
  return b0..b1..b2..b3
end

-- returns new string with word at index replaced (index is 0-based u32 index)
local function patch_word_le(buf, word_index, value)
  local byte_index = word_index * 4   -- 0-based
  local left = buf:sub(1, byte_index)
  local right = buf:sub(byte_index + 5)
  return left .. put_u32_le(value) .. right
end

----------------------------------------------------------------------
-- defaults
----------------------------------------------------------------------
local WG_SIZE = 64  -- matches WGSL @workgroup_size(64)
local HEADER_WORDS_V3 = 12
-- word 10: pp_start, word 11: pp_len

-- choose a sane entry name. If your CUDA kernel uses another name, set payload.entry in strategy.
local DEFAULT_ENTRY_BY_FW = { cuda = "execute_task", opencl = "execute_task", vulkan = "main" }

local function get_entry_point(fw, payload)
  if payload and payload.entry and #payload.entry > 0 then return payload.entry end
  return DEFAULT_ENTRY_BY_FW[norm_fw(fw)] or "execute_task"
end

----------------------------------------------------------------------
-- main entry required by your C++ LuaHost wrapper
----------------------------------------------------------------------
function compile_and_run(chunk)
  print("[lua/ecm1] compile_and_run")

  local fw = norm_fw(workload_framework or (chunk.payload and chunk.payload.framework) or chunk.framework or (chunk.meta and chunk.meta.framework) or "cuda")
  if fw ~= "cuda" then
    print("[lua/ecm1] WARNING: requested fw="..tostring(fw)..", but this host is tuned for CUDA. Proceeding anyway.")
  end

  local payload = chunk.payload or chunk
  local dims = payload.dims or {}
  local n = assert(dims.n, "payload.dims.n missing")
  local pp_count = assert(dims.pp_count, "payload.dims.pp_count missing")
  local total_words = assert(dims.total_words, "payload.dims.total_words missing")
  local bufferSizeBytes = total_words * 4

  -- pull kernel source (CUDA preferred)
  local arts = get_artifacts_view(chunk)
  local source = find_artifact_source_for(fw, arts) or find_artifact_source_for("cuda", arts)
  if not source then error("No kernel source found for CUDA in artifacts") end

  -- input buffer (as bytes); we will PATCH pp_start/pp_len each pass
  local in_b64 = assert(payload.data, "payload.data (base64) missing")
  local base_buf = b64decode(in_b64)
  if #base_buf < bufferSizeBytes then
    error(string.format("ECM buffer too small: have %d need %d bytes", #base_buf, bufferSizeBytes))
  end

  -- launch configuration (OpenCL-style fields; CudaExecutor maps them to grid/block)
  local blocks = math.floor((n + WG_SIZE - 1) / WG_SIZE) * WG_SIZE
  local local_size = { WG_SIZE, 1, 1 }
  local global_size = { blocks, 1, 1 }

  local entry = get_entry_point(fw, payload)

  -- resumable loop
  local TARGET_MS = 500
  local pp_len = math.min(1500, pp_count)
  local pp_start = 0

  local pass = 0
  local total_ms = 0
  local last_result = nil

  while pp_start < pp_count do
    local this_len = math.min(pp_len, pp_count - pp_start)

    -- patch header[10]=pp_start, header[11]=pp_len
    local buf = base_buf
    buf = patch_word_le(buf, 10, pp_start)
    buf = patch_word_le(buf, 11, this_len)

    local task = {
      action      = "compile_and_run",
      source      = source,
      entry       = entry,
      inputs      = { { b64 = b64encode(buf) } },
      outputSizes = { bufferSizeBytes }, -- read back full IO region
      global      = global_size,
    }
    task["local"] = local_size

    local t0 = os.clock()
    local result = executor.run(fw, task)
    local dt = (os.clock() - t0) * 1000.0

    if type(result) ~= "table" or not result.ok then
      local err = (type(result) == "table" and result.error) or "unknown"
      error("[lua/ecm1] executor error: " .. tostring(err))
    end
    last_result = result
    total_ms = total_ms + dt
    pass = pass + 1

    -- Adaptive pp_len (CPU time heuristic)
    if dt < TARGET_MS/2 and pp_len < 10000 then
      pp_len = math.min(pp_len * 2, 10000)
    elseif dt > TARGET_MS * 1.5 then
      local scaled = math.floor(pp_len * TARGET_MS / dt)
      pp_len = math.max(scaled, 100)
    end

    pp_start = pp_start + this_len
  end

  -- Annotate timings for your C++ client to forward in "timings"
  if type(last_result) == "table" then
    last_result.timings = last_result.timings or {}
    last_result.timings.luaPasses = pass
    last_result.timings.luaTotalMs = total_ms
    last_result.timings.wgSize = WG_SIZE
  end

  return last_result
end

print("[lua/ecm1] host loaded (resumable CUDA)")
