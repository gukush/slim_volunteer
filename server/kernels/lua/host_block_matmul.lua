-- host.lua
-- Minimal host for block matmul with NO Lua-side caching.
-- It selects the proper artifact directly from the incoming chunk,
-- and falls back to the global `artifacts` table if the runtime
-- doesn't pass artifacts into the chunk.

----------------------------------------------------------------------
-- Utilities
----------------------------------------------------------------------

local function norm_fw(fw)
  if fw == "native-opencl" then return "opencl" end
  if fw == "native-cuda"   then return "cuda"   end
  if fw == "native-vulkan" then return "vulkan" end
  return fw or "opencl"
end

-- Pure Lua Base64 decode (tolerates missing padding and stray chars)
local function b64decode(data)
  if type(data) ~= "string" or #data == 0 then return nil end
  local pad = #data % 4
  if pad > 0 then data = data .. string.rep("=", 4 - pad) end
  local b='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'
  data = data:gsub('[^'..b..'=]', '')
  return (data:gsub('.', function(x)
      if x == '=' then return '' end
      local r,f = '', (b:find(x, 1, true)-1)
      for i=6,1,-1 do r = r .. (f % 2^i - f % 2^(i-1) > 0 and '1' or '0') end
      return r
    end):gsub('%d%d%d?%d?%d?%d?%d?%d?', function(x)
      if #x ~= 8 then return '' end
      local c = 0
      for i=1,8 do if x:sub(i,i) == '1' then c = c + 2^(8-i) end end
      return string.char(c)
    end))
end

local function ensure_array(v)
  if type(v) == "table" then return v end
  return v == nil and {} or { v }
end

local function prepare_inputs(inputs)
  if type(inputs) ~= "table" then return {} end
  local out = {}
  for _, item in ipairs(inputs) do
    if type(item) == "table" then
      if item.b64 or item.bytes then
        table.insert(out, item)                -- already prepared
      elseif item.data and type(item.data) == "string" then
        table.insert(out, { b64 = item.data }) -- base64 string
      else
        table.insert(out, item)                -- passthrough
      end
    else
      table.insert(out, item)
    end
  end
  return out
end

local function round_up(n, m) return math.floor((n + m - 1) / m) * m end

local function calculate_local_size(payload, fw, rows, cols)
  if payload and payload["local"] then return payload["local"] end
  fw = norm_fw(fw)
  if fw == "opencl" then
    -- Match kernel’s TILE (16 in your .cl). If you later pass -DTILE=..., read it here.
    local tile = 16
    local lx = (cols and cols < tile) and cols or tile
    local ly = (rows and rows < tile) and rows or tile
    return { lx, ly, 1 }
  end
  return nil
end

local function get_entry_point(fw, payload)
  if payload and payload.entry and #payload.entry > 0 then return payload.entry end
  if payload and payload.program and #payload.program > 0 then return payload.program end
  local defaults = { opencl = "execute_task", cuda = "execute_task", vulkan = "main" }
  return defaults[norm_fw(fw)] or "main"
end

-- Prefer chunk.artifacts; fall back to global `artifacts`
local function get_artifacts_view(chunk)
  local arr = {}
  local function append_all(t)
    if type(t) ~= "table" then return end
    for i, a in ipairs(t) do arr[#arr+1] = a end
  end
  if chunk and type(chunk.artifacts) == "table" and #chunk.artifacts > 0 then
    append_all(chunk.artifacts)
  elseif type(artifacts) == "table" and #artifacts > 0 then
    append_all(artifacts)
  end
  return arr
end

local function list_available_backends(arts)
  local seen, out = {}, {}
  for _, a in ipairs(arts) do
    local b = norm_fw(a.backend)
    if b and not seen[b] then
      seen[b] = true
      out[#out+1] = b
    end
  end
  return out
end

-- Pull kernel source for the current framework from artifacts view
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

-- Global size: for OpenCL use 2-D NDRange {cols, rows, 1}; fallback otherwise
local function calculate_global_size(payload, fw, rows, cols)
  if payload and payload.global then return payload.global end
  fw = norm_fw(fw)
  if rows and cols and fw == "opencl" then
    return { cols, rows, 1 }
  end
  local outSizes = (payload and payload.outputSizes) or {}
  if #outSizes > 0 then
    local floats = math.max(1, math.ceil(outSizes[1] / 4))
    return floats
  end
  return 1
end

----------------------------------------------------------------------
-- Main entry
----------------------------------------------------------------------

function compile_and_run(chunk)
  print("[lua] compile_and_run called")

  local fw = norm_fw(chunk.framework or (chunk.meta and chunk.meta.framework) or "opencl")
  local payload = chunk.payload or chunk
  local cfg = chunk.config or payload.config or {}

  print("[lua] Framework: " .. tostring(fw) .. ", Action: " .. tostring(payload.action or "compile_and_run"))

  -- Uniforms: rows (M), K, cols (N)
  local uniforms = payload.uniforms
  local M = cfg.M or cfg.rows or payload.M
  local K = cfg.K or payload.K
  local N = cfg.N or cfg.cols or payload.N
  if (not uniforms or #uniforms < 3) and (M and K and N) then
    uniforms = { M, K, N }
  end
  uniforms = ensure_array(uniforms)
  local rows = uniforms[1]
  local cols = uniforms[3]

  -- Entry
  local entry = get_entry_point(fw, payload)

  -- Artifacts view (chunk or global)
  local arts = get_artifacts_view(chunk)
  local source = find_artifact_source_for(fw, arts)
  if not source then
    local avail = table.concat(list_available_backends(arts), ", ")
    error("No kernel source found in chunk/global artifacts for framework '" ..
          tostring(fw) .. "'. Available backends: " .. ( #avail > 0 and avail or "none" ))
  end
  print(string.format("[lua] Using artifact source for %s (%d chars), entry='%s'", fw, #source, entry))

  -- Local & global sizes
  local local_ = calculate_local_size(payload, fw, rows, cols)
  local global = calculate_global_size(payload, fw, rows, cols)
  if type(global) == "table" and local_ then
  -- OpenCL requires global to be a multiple of local in each dim
  global[1] = round_up(global[1], local_[1] or 1)
  global[2] = round_up(global[2], local_[2] or 1)
end
  print("[lua] Global size: " ..
        (type(global) == "table" and ("{"..table.concat(global, ", ").."}") or tostring(global)))

  -- Output sizes (float32 MxN) if missing
  local outputSizes = payload.outputSizes
  if (not outputSizes or #outputSizes == 0) and rows and cols then
    outputSizes = { rows * cols * 4 }
  end

  -- Build task for executor
  local task = {
    action      = payload.action or "compile_and_run",
    source      = source,
    entry       = entry,
    inputs      = prepare_inputs(payload.inputs),
    uniforms    = uniforms,
    outputSizes = ensure_array(outputSizes),
    global      = global,
  }
  -- due to local being reserved keyword in lua
  task["local"] = local_

  if payload.buildOptions then task.buildOptions = payload.buildOptions end
  if payload.defines      then task.defines      = payload.defines      end

  print("[lua] Calling executor: " .. fw)
  print("[lua]   Entry: " .. tostring(task.entry))
  print("[lua]   Uniforms: " .. (task.uniforms and ("["..table.concat(task.uniforms, ", ").."]") or "nil"))
  print("[lua]   Outputs: " .. (#task.outputSizes > 0 and ("["..table.concat(task.outputSizes, ", ").."]") or "nil"))

  local result = executor.run(fw, task)

  print("[lua] Executor ok=" .. tostring(result and result.ok))
  if result and not result.ok then
    print("[lua] Error: " .. tostring(result.error or "unknown"))
  end

  if type(result) == "table" then
    return result
  end
  return { ok = false, error = "Executor returned unexpected type: " .. type(result) }
end

print("[lua] host.lua loaded (no Lua-side kernel caching; chunk/global artifacts supported).")
