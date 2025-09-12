-- host_block_matmul.lua
-- Lua host that mirrors native-bridge.client.js behavior for block matmul
-- Works with CUDA and OpenCL; Vulkan stub included for symmetry.

-- Helper: find an artifact by filename suffix; decode base64 if needed
local function find_artifact_by_ext(ext)
  if not workload or not workload.artifacts then return nil end
  for _, a in ipairs(workload.artifacts) do
    if a.name and a.name:match(ext .. "$") then
      local bytes = a.bytes
      if a.encoding == "base64" and type(bytes) == "string" then
        local ok, decoded = pcall(b64.decode, bytes)
        if ok and decoded then return decoded end
      end
      return bytes
    end
  end
  return nil
end

local function pick_kernel(framework)
  if framework == "cuda" then
    return find_artifact_by_ext("%.cu")
  elseif framework == "opencl" then
    return find_artifact_by_ext("%.cl")
  elseif framework == "vulkan" then
    -- prefer SPIR-V if you ship it: .spv.b64, else GLSL
    return find_artifact_by_ext("%.glsl")
  end
  return nil
end

local function ensure_array(x)
  if type(x) == "table" then return x end
  return {}
end

function compile_and_run(chunk)
  local meta    = chunk.meta    or {}
  local payload = chunk.payload or {}

  -- Accept both meta.framework and meta.backend
  local fw = meta.framework or meta.backend or "opencl"
  fw = tostring(fw):gsub("^native%-", ""):lower()

  -- Pull uniforms and inputs (buffers are already base64 strings from server)
  local uniforms = payload.uniforms or meta.uniforms or {}
  local inputs = {}
  if type(payload.buffers) == "table" then
    for _, b in ipairs(payload.buffers) do
      if type(b) == "string" then table.insert(inputs, { data = b }) end
    end
  else
    if type(payload.a) == "string" then table.insert(inputs, { data = payload.a }) end
    if type(payload.b) == "string" then table.insert(inputs, { data = payload.b }) end
    if type(payload.data) == "string" then table.insert(inputs, { data = payload.data }) end
  end

  local outputSizes = meta.outputSizes or payload.outputSizes or { 1024 }

  -- Get kernel source
  local source = payload.source or pick_kernel(fw)
  assert(type(source) == "string" and #source > 0, "kernel source missing for framework "..fw)

  -- Default entry (change if your kernels use different symbol)
  local entry = meta.entry or payload.entry or "execute_task"

  -- Dispatch parameters
  local task = {
    source       = source,
    entry        = entry,
    uniforms     = uniforms,
    inputs       = inputs,
    outputSizes  = outputSizes,
  }

  if fw == "cuda" then
    local d = (meta.dispatch and meta.dispatch.cuda) or {}
    local rows = tonumber(meta.rows or 1)
    local cols = tonumber(meta.cols or 1)
    task.grid  = ensure_array(d.grid or { math.ceil(cols/16), math.ceil(rows/16), 1 })
    task.block = ensure_array(d.block or { 16, 16, 1 })
  elseif fw == "opencl" then
    local d = (meta.dispatch and meta.dispatch.opencl) or {}
    local rows = tonumber(meta.rows or 1)
    local cols = tonumber(meta.cols or 1)
    task.global = ensure_array(d.global or { cols, rows, 1 })
    task.local  = ensure_array(d.local  or { 16, 16, 1 })
  elseif fw == "vulkan" then
    -- If you add Vulkan executor, accept GLSL or SPIR-V and threadgroup dims here
    local d = (meta.dispatch and meta.dispatch.vulkan) or {}
    task.groups = ensure_array(d.groups or { 1, 1, 1 })
  else
    error("Unsupported framework: "..tostring(fw))
  end

  -- Call into the native executor via C++ binding
  local result = executor.run(fw, task)

  -- Normalize: return { result = <base64> } shape for the C++ wrapper
  if type(result) == "table" then
    if type(result.result) == "string" then return result end
    if type(result.result_b64) == "string" then return { result = result.result_b64 } end
    if type(result.results) == "table" and #result.results > 0 then
      return { result = result.results[1] }
    end
  end
  return result
end
