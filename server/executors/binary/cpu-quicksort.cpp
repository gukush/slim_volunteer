#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdint>
#include "../kernels/exe/common_protocol.h"

// Simple quicksort implementation for CPU
template<typename T>
void quicksort(T* arr, int left, int right, bool ascending = true) {
    if (left < right) {
        // Choose pivot as middle element
        int pivot = left + (right - left) / 2;
        T pivotValue = arr[pivot];

        // Partition around pivot
        int i = left, j = right;
        while (i <= j) {
            if (ascending) {
                while (arr[i] < pivotValue) i++;
                while (arr[j] > pivotValue) j--;
            } else {
                while (arr[i] > pivotValue) i++;
                while (arr[j] < pivotValue) j--;
            }

            if (i <= j) {
                std::swap(arr[i], arr[j]);
                i++;
                j--;
            }
        }

        // Recursively sort partitions
        quicksort(arr, left, j, ascending);
        quicksort(arr, i, right, ascending);
    }
}

// Read binary data from stdin
std::vector<uint32_t> readBinaryFromStdin() {
    std::vector<uint32_t> data;
    std::vector<char> buffer(4096); // Read in chunks

    while (std::cin.read(buffer.data(), buffer.size()) || std::cin.gcount() > 0) {
        size_t bytesRead = std::cin.gcount();

        // Convert bytes to uint32_t values
        for (size_t i = 0; i < bytesRead; i += 4) {
            if (i + 3 < bytesRead) {
                uint32_t value = 0;
                value |= static_cast<uint8_t>(buffer[i]);
                value |= static_cast<uint8_t>(buffer[i + 1]) << 8;
                value |= static_cast<uint8_t>(buffer[i + 2]) << 16;
                value |= static_cast<uint8_t>(buffer[i + 3]) << 24;
                data.push_back(value);
            }
        }
    }

    return data;
}

// Write binary data to stdout
void writeBinaryToStdout(const std::vector<uint32_t>& data) {
    std::cout.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(uint32_t));
    std::cout.flush();
}

// Read binary file and return as vector (fallback for file mode)
std::vector<uint32_t> readBinaryFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    // Get file size
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    // Check if size is multiple of 4 (uint32_t)
    if (fileSize % 4 != 0) {
        throw std::runtime_error("File size is not a multiple of 4 bytes");
    }

    // Read data
    std::vector<uint32_t> data(fileSize / 4);
    file.read(reinterpret_cast<char*>(data.data()), fileSize);

    if (!file) {
        throw std::runtime_error("Error reading file");
    }

    return data;
}

// Write binary file (fallback for file mode)
void writeBinaryFile(const std::string& filename, const std::vector<uint32_t>& data) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot create file: " + filename);
    }

    file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(uint32_t));

    if (!file) {
        throw std::runtime_error("Error writing file");
    }
}

int main(int argc, char* argv[]) {
    // Check for stdin/stdout mode (no arguments or special flag)
    bool useStreams = (argc == 1) || (argc == 2 && std::string(argv[1]) == "--stdin");

    if (useStreams) {
        // Stream mode: read from stdin, write to stdout using common protocol
        try {
            EXE_LOG_INFO("Starting CPU quicksort");

            // Read all data from stdin
            std::vector<uint8_t> input_buffer;
            const size_t CHUNK_SIZE = 1 << 20; // 1MB chunks
            uint8_t chunk[CHUNK_SIZE];
            size_t bytes_read;

            while ((bytes_read = std::fread(chunk, 1, CHUNK_SIZE, stdin)) > 0) {
                input_buffer.insert(input_buffer.end(), chunk, chunk + bytes_read);
            }

            if (std::ferror(stdin)) {
                EXE_LOG_ERROR("Error reading from stdin");
                return 1;
            }

            if (input_buffer.empty()) {
                EXE_LOG_ERROR("No data received from stdin");
                return 1;
            }

            EXE_LOG_INFO("Received " << input_buffer.size() << " bytes from stdin");

            // Parse protocol header
            ProtocolHeader header;
            const uint8_t* data = input_buffer.data();
            size_t remaining = input_buffer.size();

            EXE_VALIDATE(ExeProtocol::readHeader(data, remaining, header),
                         "Failed to read protocol header");

            EXE_LOG_INFO("Protocol version: " << header.version);
            EXE_LOG_INFO("Framework: " << ExeProtocol::getFrameworkName(static_cast<FrameworkType>(header.framework)));
            EXE_LOG_INFO("Data type: " << ExeProtocol::getDataTypeName(static_cast<DataType>(header.data_type)));
            EXE_LOG_INFO("Inputs: " << header.num_inputs << ", Outputs: " << header.num_outputs);

            // Skip metadata if present
            if (header.metadata_size > 0) {
                EXE_VALIDATE(remaining >= header.metadata_size, "Insufficient data for metadata");
                data += header.metadata_size;
                remaining -= header.metadata_size;
            }

            // Parse input buffers
            EXE_VALIDATE(header.num_inputs >= 1, "Expected at least 1 input buffer");

            std::vector<std::vector<uint8_t>> input_buffers;
            std::vector<BufferDescriptor> input_descriptors;

            for (uint32_t i = 0; i < header.num_inputs; ++i) {
                BufferDescriptor desc;
                std::vector<uint8_t> buffer;

                EXE_VALIDATE(ExeProtocol::readBuffer(data, remaining, desc, buffer),
                             "Failed to read input buffer " << i);

                input_descriptors.push_back(desc);
                input_buffers.push_back(std::move(buffer));

                EXE_LOG_DEBUG("Input " << i << ": " << desc.size << " bytes, dims=["
                             << desc.dimensions[0] << "," << desc.dimensions[1]
                             << "," << desc.dimensions[2] << "," << desc.dimensions[3] << "]");
            }

            // Extract sorting parameters from first input buffer
            EXE_VALIDATE(input_descriptors[0].data_type == static_cast<uint32_t>(DataType::UINT32),
                         "Input data must be uint32");

            const uint32_t* input_data = reinterpret_cast<const uint32_t*>(input_buffers[0].data());
            size_t num_elements = input_buffers[0].size() / sizeof(uint32_t);

            EXE_LOG_INFO("Sorting " << num_elements << " integers");

            // Create working copy
            std::vector<uint32_t> data_to_sort(input_data, input_data + num_elements);

            // Determine sort direction (default ascending)
            bool ascending = true;
            // Could extract from metadata or use a parameter, for now use default

            // Measure sorting time
            auto start = std::chrono::high_resolution_clock::now();
            quicksort(data_to_sort.data(), 0, data_to_sort.size() - 1, ascending);
            auto end = std::chrono::high_resolution_clock::now();

            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            double ms = duration.count() / 1000.0;

            EXE_LOG_INFO("Sort completed in " << ms << " ms");

            // Write result to stdout
            writeBinaryToStdout(data_to_sort);

            EXE_LOG_INFO("CPU quicksort completed successfully");

            return 0;

        } catch (const std::exception& e) {
            EXE_LOG_ERROR("Error: " << e.what());
            return 1;
        }
    } else {
        // File mode: original behavior
        if (argc < 4) {
            std::cerr << "Usage: " << argv[0] << " [--stdin] | <input_file> <output_file> <ascending> [original_size]" << std::endl;
            std::cerr << "  --stdin: Read from stdin, write to stdout" << std::endl;
            std::cerr << "  input_file: Path to input binary file containing uint32_t integers" << std::endl;
            std::cerr << "  output_file: Path to output binary file" << std::endl;
            std::cerr << "  ascending: 1 for ascending sort, 0 for descending" << std::endl;
            std::cerr << "  original_size: (optional) Number of original elements to keep in output" << std::endl;
            return 1;
        }

        try {
            std::string inputFile = argv[1];
            std::string outputFile = argv[2];
            bool ascending = (std::stoi(argv[3]) != 0);
            int originalSize = -1;

            if (argc > 4) {
                originalSize = std::stoi(argv[4]);
            }

            std::cout << "[cpu-quicksort] Reading input file: " << inputFile << std::endl;
            auto data = readBinaryFile(inputFile);
            std::cout << "[cpu-quicksort] Loaded " << data.size() << " integers" << std::endl;

            // Determine actual size to sort
            int sortSize = (originalSize > 0 && originalSize < static_cast<int>(data.size()))
                          ? originalSize : static_cast<int>(data.size());

            std::cout << "[cpu-quicksort] Sorting " << sortSize << " integers (ascending=" << ascending << ")" << std::endl;

            // Measure sorting time
            auto start = std::chrono::high_resolution_clock::now();
            quicksort(data.data(), 0, sortSize - 1, ascending);
            auto end = std::chrono::high_resolution_clock::now();

            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            double ms = duration.count() / 1000.0;

            std::cout << "[cpu-quicksort] Sort completed in " << ms << " ms" << std::endl;

            // Write output (only the sorted portion if originalSize was specified)
            std::vector<uint32_t> outputData(data.begin(), data.begin() + sortSize);
            writeBinaryFile(outputFile, outputData);

            std::cout << "[cpu-quicksort] Output written to: " << outputFile << std::endl;
            std::cout << "[cpu-quicksort] Output contains " << outputData.size() << " integers" << std::endl;

            // Print first few elements for verification
            std::cout << "[cpu-quicksort] First 10 elements: ";
            for (int i = 0; i < std::min(10, static_cast<int>(outputData.size())); i++) {
                std::cout << outputData[i] << " ";
            }
            std::cout << std::endl;

            return 0;

        } catch (const std::exception& e) {
            std::cerr << "[cpu-quicksort] Error: " << e.what() << std::endl;
            return 1;
        }
    }
}
