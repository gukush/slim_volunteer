NVCC ?= nvcc
ARCH ?= sm_70
CXXFLAGS ?= -O3 -std=c++17

TARGET := cuda_pollard_pminus1
SRC := cuda_pollard_pminus1.cu

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CXXFLAGS) -arch=$(ARCH) -o $@ $<

clean:
	rm -f $(TARGET)
