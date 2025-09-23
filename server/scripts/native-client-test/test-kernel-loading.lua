#!/usr/bin/env lua
-- Test script to verify kernel loading functionality
-- This script tests the kernel loading logic without requiring the full executor setup

-- Mock the json module for testing
local json = {
    encode = function(t) return "{}" end,
    decode = function(s) return {} end
}

-- Mock the executor for testing
local mock_executor = {
    run = function(framework, task)
        print("Mock executor called with framework:", framework)
        print("Task source length:", task.source and #task.source or 0)
        print("Task entry:", task.entry)
        print("Task uniforms:", table.concat(task.uniforms or {}, ", "))
        return {
            ok = true,
            outputs = {{1, 2, 3, 4}}, -- Mock output
            timings = {compileMs = 0.1, kernelMs = 0.5}
        }
    end
}

-- Load the host script
dofile("executors/host_block_matmul.lua")

-- Test kernel loading for each framework
local frameworks = {"cuda", "opencl", "vulkan"}

print("Testing kernel loading functionality...")
print("=" .. string.rep("=", 50))

for _, framework in ipairs(frameworks) do
    print("\nTesting " .. framework .. " framework:")
    print("-" .. string.rep("-", 30))

    local success, result = pcall(function()
        -- Test kernel loading
        local kernel_source = get_kernel_source(framework)
        print("✓ Kernel loaded successfully")
        print("  Source length:", #kernel_source)
        print("  Contains 'execute_task':", kernel_source:find("execute_task") and "Yes" or "No")

        -- Test compile_and_run with mock data
        local mock_chunk = {
            meta = { framework = framework },
            payload = {
                dims = { rows = 4, K = 4, cols = 4 },
                a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}, -- 4x4 matrix
                b = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}  -- 4x4 matrix
            }
        }

        local result = compile_and_run(mock_chunk)
        print("✓ compile_and_run completed")
        print("  Result ok:", result.ok)
        print("  Result length:", result.result and #result.result or 0)

        return result
    end)

    if success then
        print("✓ " .. framework .. " test passed")
    else
        print(" " .. framework .. " test failed:", result)
    end
end

print("\n" .. string.rep("=", 50))
print("Kernel loading test completed!")
