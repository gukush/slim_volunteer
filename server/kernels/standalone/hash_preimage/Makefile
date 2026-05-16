# Makefile for standalone CUDA hash preimage search

NVCC = nvcc
CXX_FLAGS = -O3 -std=c++17
CUDA_ARCH ?= -arch=sm_70

SRC = hash_preimage_cuda.cu
TARGET = hash-preimage-search

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CXX_FLAGS) $(CUDA_ARCH) -o $@ $<

clean:
	rm -f $(TARGET)

.PHONY: all clean
