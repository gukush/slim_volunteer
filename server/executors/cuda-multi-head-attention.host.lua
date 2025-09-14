-- File: executors/cuda-multi-head-attention.host.lua
-- Minimal Lua host for single-head attention on CUDA.
-- Expects a CUDA source artifact in the chunk/global artifacts
-- with backend="cuda" and entry "execute_task".
--
-- Uniforms layout: [seq_len, d_k, d_v]
-- Inputs: Q (seq_len*d_k floats), K (seq_len*d_k floats), V (seq_len*d_v floats)
-- Output: seq_len*d_v floats

----------------------------------------------------------------------
-- helpers
----------------------------------------------------------------------

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

local function ensure_array(v) return (type(v) == "table") and v or (v == nil and {} or { v }) end

local function get_artifacts_view(chunk)
  local out = {}
  local function add(list)
    if type(list) ~= "table" then return end
    for _, a in ipairs(list) do out[#out+1] = a end
  end
  if chunk and type(chunk.artifacts) == "table" and #chunk.artifacts > 0 then
    add(chunk.artifacts)
  elseif type(artifacts) == "table" and #artifacts > 0 then
    add(artifacts)
  end
  return out
end

local function find_cuda_source(arts)
  for _, a in ipairs(arts) do
    if a and a.type == "text" and (a.backend == "cuda" or a.backend == "native-cuda") and a.bytes then
      local src = b64decode(a.bytes)
      if type(src) == "string" and #src > 0 then return src end
    end
  end
  return nil
end

----------------------------------------------------------------------
-- main
----------------------------------------------------------------------

function compile_and_run(chunk)
  print("[lua][mha-cuda] compile_and_run")

  local payload = chunk.payload or chunk
  local cfg = chunk.config or payload.config or {}

  -- infer uniforms (seq_len, d_k, d_v)
  local u = ensure_array(payload.uniforms)
  local seq_len = u[1] or cfg.seq_len or payload.seq_len
  local d_k     = u[2] or cfg.d_k     or payload.d_k
  local d_v     = u[3] or cfg.d_v     or payload.d_v
  if not (seq_len and d_k and d_v) then
    error("[lua][mha-cuda] missing uniforms: need seq_len, d_k, d_v")
  end
  local uniforms = { seq_len, d_k, d_v }

  -- collect CUDA source from artifacts
  local arts = get_artifacts_view(chunk)
  local source = find_cuda_source(arts)
  if not source then
    error("[lua][mha-cuda] no CUDA source artifact (backend='cuda') found")
  end

  -- IO sizes
  local outBytes = seq_len * d_v * 4
  local outputSizes = { outBytes }

  -- CUDA launch: one block per (row, feature), 256 threads reduce across seq_len
  local grid  = payload.grid  or { seq_len, d_v, 1 }
  local block = payload.block or { 256, 1, 1 }

  -- Build executor task
  local task = {
    action      = "compile_and_run",
    source      = source,
    entry       = payload.entry or "execute_task",
    inputs      = payload.inputs,  -- expected as raw binary chunks already prepared upstream
    uniforms    = uniforms,
    outputSizes = outputSizes,
    grid        = grid,
    block       = block,
    defines     = payload.defines,
    buildOptions= payload.buildOptions,
  }

  print(string.format("[lua][mha-cuda] grid={%d,%d,%d} block={%d,%d,%d}",
    grid[1],grid[2],grid[3], block[1],block[2],block[3]))
  print(string.format("[lua][mha-cuda] uniforms=[%d,%d,%d] -> out %d bytes",
    uniforms[1],uniforms[2],uniforms[3], outBytes))

  local result = executor.run("cuda", task)
  if type(result) ~= "table" then
    return { ok=false, error="executor returned non-table" }
  end
  return result
end

print("[lua][mha-cuda] host loaded")
