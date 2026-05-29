NVCC ?= nvcc
ARCH ?= sm_70
CXXFLAGS ?= -O3 -std=c++17

TARGET := cuda_torus_grid_rl
SRC := cuda_torus_grid_rl.cu

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CXXFLAGS) -arch=$(ARCH) -o $@ $<

run: $(TARGET)
	./$(TARGET) --totalTrajectories=65536 --chunkSize=4096 --blockSize=128

clean:
	rm -f $(TARGET)
