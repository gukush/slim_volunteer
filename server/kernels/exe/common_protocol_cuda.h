// common_protocol_cuda.h
// CUDA-compatible version of the common protocol header

#ifndef COMMON_PROTOCOL_CUDA_H
#define COMMON_PROTOCOL_CUDA_H

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// Protocol constants
#define EXE_PROTOCOL_MAGIC 0x4558454D  // "EXEM"
#define EXE_PROTOCOL_VERSION 1

// Framework types
enum class FrameworkType : uint32_t {
    CPU = 0,
    CUDA = 1,
    OPENCL = 2,
    VULKAN = 3,
    WEBGPU = 4
};

// Data types
enum class DataType : uint32_t {
    FLOAT32 = 0,
    FLOAT16 = 1,
    INT32 = 2,
    INT16 = 3,
    INT8 = 4,
    UINT32 = 5,
    UINT16 = 6,
    UINT8 = 7
};

// Protocol structures
struct ProtocolHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t framework;
    uint32_t data_type;
    uint32_t num_inputs;
    uint32_t num_outputs;
    uint32_t metadata_size;
    uint32_t reserved;
};

struct BufferDescriptor {
    uint32_t size;
    uint32_t data_type;
    uint32_t dimensions[4];
    uint32_t reserved[3];
};

// Utility functions
class ExeProtocol {
public:
    static bool readHeader(const uint8_t*& data, size_t& remaining, ProtocolHeader& header) {
        if (remaining < sizeof(ProtocolHeader)) {
            return false;
        }

        memcpy(&header, data, sizeof(ProtocolHeader));
        data += sizeof(ProtocolHeader);
        remaining -= sizeof(ProtocolHeader);

        if (header.magic != EXE_PROTOCOL_MAGIC) {
            return false;
        }

        if (header.version != EXE_PROTOCOL_VERSION) {
            return false;
        }

        return true;
    }

    static bool readBuffer(const uint8_t*& data, size_t& remaining, BufferDescriptor& desc, std::vector<uint8_t>& buffer) {
        if (remaining < sizeof(BufferDescriptor)) {
            return false;
        }

        memcpy(&desc, data, sizeof(BufferDescriptor));
        data += sizeof(BufferDescriptor);
        remaining -= sizeof(BufferDescriptor);

        if (remaining < desc.size) {
            return false;
        }

        buffer.resize(desc.size);
        memcpy(buffer.data(), data, desc.size);
        data += desc.size;
        remaining -= desc.size;

        return true;
    }

    static const char* getFrameworkName(FrameworkType framework) {
        switch (framework) {
            case FrameworkType::CPU: return "CPU";
            case FrameworkType::CUDA: return "CUDA";
            case FrameworkType::OPENCL: return "OpenCL";
            case FrameworkType::VULKAN: return "Vulkan";
            case FrameworkType::WEBGPU: return "WebGPU";
            default: return "Unknown";
        }
    }

    static const char* getDataTypeName(DataType dataType) {
        switch (dataType) {
            case DataType::FLOAT32: return "Float32";
            case DataType::FLOAT16: return "Float16";
            case DataType::INT32: return "Int32";
            case DataType::INT16: return "Int16";
            case DataType::INT8: return "Int8";
            case DataType::UINT32: return "Uint32";
            case DataType::UINT16: return "Uint16";
            case DataType::UINT8: return "Uint8";
            default: return "Unknown";
        }
    }
};

// Logging macros
#define EXE_LOG_INFO(msg) do { \
    std::fprintf(stderr, "[INFO] %s\n", msg); \
    std::fflush(stderr); \
} while(0)

#define EXE_LOG_ERROR(msg) do { \
    std::fprintf(stderr, "[ERROR] %s\n", msg); \
    std::fflush(stderr); \
} while(0)

#define EXE_LOG_DEBUG(msg) do { \
    std::fprintf(stderr, "[DEBUG] %s\n", msg); \
    std::fflush(stderr); \
} while(0)

#define EXE_VALIDATE(condition, message) do { \
    if (!(condition)) { \
        EXE_LOG_ERROR(message); \
        std::exit(1); \
    } \
} while(0)

#endif // COMMON_PROTOCOL_CUDA_H
