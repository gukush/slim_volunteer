// common_protocol.h
// Standardized protocol for all exe kernels to ensure consistency
// This allows the binary executor to be completely task-agnostic

#ifndef COMMON_PROTOCOL_H
#define COMMON_PROTOCOL_H

#include <cstdint>
#include <vector>
#include <string>
#include <cstring>
#include <stdexcept>

// Protocol version for compatibility checking
#define EXE_PROTOCOL_VERSION 1

// Magic number to identify the protocol
#define EXE_MAGIC 0x4558454D  // "EXEM"

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

// Protocol header structure
struct ProtocolHeader {
    uint32_t magic;           // EXE_MAGIC
    uint32_t version;         // EXE_PROTOCOL_VERSION
    uint32_t framework;       // FrameworkType
    uint32_t data_type;       // DataType
    uint32_t num_inputs;      // Number of input buffers
    uint32_t num_outputs;     // Number of output buffers
    uint32_t metadata_size;   // Size of metadata section
    uint32_t reserved;        // Reserved for future use
};

// Buffer descriptor
struct BufferDescriptor {
    uint32_t size;            // Size in bytes
    uint32_t data_type;       // DataType
    uint32_t dimensions[4];   // Up to 4D dimensions (0 = unused)
    uint32_t reserved[3];     // Reserved for future use
};

// Utility functions for reading/writing the protocol
class ExeProtocol {
public:
    static void writeHeader(std::vector<uint8_t>& buffer, 
                           FrameworkType framework, 
                           DataType data_type,
                           uint32_t num_inputs, 
                           uint32_t num_outputs,
                           const std::string& metadata = "") {
        ProtocolHeader header;
        header.magic = EXE_MAGIC;
        header.version = EXE_PROTOCOL_VERSION;
        header.framework = static_cast<uint32_t>(framework);
        header.data_type = static_cast<uint32_t>(data_type);
        header.num_inputs = num_inputs;
        header.num_outputs = num_outputs;
        header.metadata_size = static_cast<uint32_t>(metadata.size());
        header.reserved = 0;

        // Write header
        const uint8_t* header_bytes = reinterpret_cast<const uint8_t*>(&header);
        buffer.insert(buffer.end(), header_bytes, header_bytes + sizeof(ProtocolHeader));

        // Write metadata if any
        if (!metadata.empty()) {
            buffer.insert(buffer.end(), metadata.begin(), metadata.end());
        }
    }

    static void writeBuffer(std::vector<uint8_t>& buffer, 
                           const void* data, 
                           size_t size, 
                           DataType data_type,
                           const std::vector<uint32_t>& dimensions = {}) {
        BufferDescriptor desc;
        desc.size = static_cast<uint32_t>(size);
        desc.data_type = static_cast<uint32_t>(data_type);
        
        // Copy dimensions (pad with zeros if less than 4)
        for (size_t i = 0; i < 4; ++i) {
            desc.dimensions[i] = (i < dimensions.size()) ? dimensions[i] : 0;
        }
        
        // Zero out reserved fields
        for (size_t i = 0; i < 3; ++i) {
            desc.reserved[i] = 0;
        }

        // Write descriptor
        const uint8_t* desc_bytes = reinterpret_cast<const uint8_t*>(&desc);
        buffer.insert(buffer.end(), desc_bytes, desc_bytes + sizeof(BufferDescriptor));

        // Write data
        const uint8_t* data_bytes = static_cast<const uint8_t*>(data);
        buffer.insert(buffer.end(), data_bytes, data_bytes + size);
    }

    static bool readHeader(const uint8_t* data, size_t size, ProtocolHeader& header) {
        if (size < sizeof(ProtocolHeader)) {
            return false;
        }

        std::memcpy(&header, data, sizeof(ProtocolHeader));
        
        if (header.magic != EXE_MAGIC) {
            return false;
        }
        
        if (header.version != EXE_PROTOCOL_VERSION) {
            return false;
        }
        
        return true;
    }

    static bool readBuffer(const uint8_t*& data, size_t& remaining, 
                          BufferDescriptor& desc, std::vector<uint8_t>& buffer) {
        if (remaining < sizeof(BufferDescriptor)) {
            return false;
        }

        std::memcpy(&desc, data, sizeof(BufferDescriptor));
        data += sizeof(BufferDescriptor);
        remaining -= sizeof(BufferDescriptor);

        if (remaining < desc.size) {
            return false;
        }

        buffer.assign(data, data + desc.size);
        data += desc.size;
        remaining -= desc.size;

        return true;
    }

    static std::string getFrameworkName(FrameworkType framework) {
        switch (framework) {
            case FrameworkType::CPU: return "CPU";
            case FrameworkType::CUDA: return "CUDA";
            case FrameworkType::OPENCL: return "OpenCL";
            case FrameworkType::VULKAN: return "Vulkan";
            case FrameworkType::WEBGPU: return "WebGPU";
            default: return "Unknown";
        }
    }

    static std::string getDataTypeName(DataType data_type) {
        switch (data_type) {
            case DataType::FLOAT32: return "float32";
            case DataType::FLOAT16: return "float16";
            case DataType::INT32: return "int32";
            case DataType::INT16: return "int16";
            case DataType::INT8: return "int8";
            case DataType::UINT32: return "uint32";
            case DataType::UINT16: return "uint16";
            case DataType::UINT8: return "uint8";
            default: return "unknown";
        }
    }
};

// Helper macros for common operations
#define EXE_LOG_INFO(msg) std::cerr << "[EXE] " << msg << std::endl
#define EXE_LOG_ERROR(msg) std::cerr << "[EXE] ERROR: " << msg << std::endl
#define EXE_LOG_DEBUG(msg) std::cerr << "[EXE] DEBUG: " << msg << std::endl

#define EXE_VALIDATE(condition, msg) \
    do { \
        if (!(condition)) { \
            EXE_LOG_ERROR(msg); \
            return 1; \
        } \
    } while(0)

#endif // COMMON_PROTOCOL_H
