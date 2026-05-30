NVCC ?= nvcc
ARCH ?= sm_70
CXXFLAGS ?= -O3 -std=c++17

TARGET := cuda_torus_grid_rl
CONST_STEPS_TARGET := cuda_torus_grid_rl_const_steps
SRC := cuda_torus_grid_rl.cu
CONST_STEPS_SRC := cuda_torus_grid_rl_const_steps.cu

.PHONY: all const-steps clean run

all: $(TARGET)

const-steps: $(CONST_STEPS_TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CXXFLAGS) -arch=$(ARCH) -o $@ $<

$(CONST_STEPS_TARGET): $(CONST_STEPS_SRC)
	$(NVCC) $(CXXFLAGS) -arch=$(ARCH) -o $@ $<

run: $(TARGET)
	./$(TARGET) --totalTrajectories=65536 --chunkSize=4096 --blockSize=128

clean:
	rm -f $(TARGET) $(CONST_STEPS_TARGET)
