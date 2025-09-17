# Makefile for Out-of-Core Matrix Multiplication

# Compiler and flags
NVCC = nvcc
CXX_FLAGS = -O3 -std=c++17
CUDA_ARCH = -arch=sm_70 # Volta or newer recommended
LIBS = -lcublas

# Source and target
SRC = out_of_core_matmul.cu
TARGET_CUSTOM = matmul-custom
TARGET_CUBLAS = matmul-cublas

# Default target
all: $(TARGET_CUSTOM) $(TARGET_CUBLAS)

# Build the custom kernel version
$(TARGET_CUSTOM): $(SRC)
	$(NVCC) $(CXX_FLAGS) $(CUDA_ARCH) -o $@ $<

# Build the cuBLAS version
$(TARGET_CUBLAS): $(SRC)
	$(NVCC) $(CXX_FLAGS) $(CUDA_ARCH) $(LIBS) -o $@ $<

# Clean up build artifacts
clean:
	rm -f $(TARGET_CUSTOM) $(TARGET_CUBLAS)

# Phony targets
.PHONY: all clean
