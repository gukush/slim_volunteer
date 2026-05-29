# Makefile for GridPack-2D RL standalone CUDA and MPI builds
#
# Adjust ARCH to match your GPU (e.g. sm_70 for V100, sm_80 for A100, sm_89 for RTX 4090)
ARCH ?= sm_70

# Maximum MLP architecture caps (must be >= any runtime value used)
MAX_STATE_DIM  ?= 8
MAX_HIDDEN_DIM ?= 128
MAX_ACTION_DIM ?= 256

# CUDA compiler
NVCC ?= nvcc

# MPI compiler wrapper (usually mpicxx or mpicc)
MPICXX ?= mpicxx

# Common flags
CUDA_FLAGS = -O3 -arch=$(ARCH) -DMAX_STATE_DIM=$(MAX_STATE_DIM) -DMAX_HIDDEN_DIM=$(MAX_HIDDEN_DIM) -DMAX_ACTION_DIM=$(MAX_ACTION_DIM)
CXX_FLAGS  = -std=c++17

# -----------------------------------------------------------------------------
# Standalone single-GPU target
# -----------------------------------------------------------------------------
standalone: gridpack-2d-rl-cuda

gridpack-2d-rl-cuda: gridpack_2d_rl_cuda.cu
	$(NVCC) $(CUDA_FLAGS) $(CXX_FLAGS) -o $@ $<

# -----------------------------------------------------------------------------
# MPI multi-GPU / multi-node target
# -----------------------------------------------------------------------------
mpi: gridpack-2d-rl-mpi

gridpack-2d-rl-mpi: gridpack_2d_rl_mpi.cu
	$(MPICXX) -x cu $(CUDA_FLAGS) $(CXX_FLAGS) -o $@ $<

# -----------------------------------------------------------------------------
# Convenience: generate random theta + epsilon for testing (uses runtime dims)
# -----------------------------------------------------------------------------
theta.bin:
	python3 -c "import numpy as np,sys; \
	  dim=8; hid=32; act=64; \
	  sz=dim*hid+hid+hid*act+act; \
	  np.random.normal(0,np.sqrt(2.0/(dim+hid)),sz).astype('float32').tofile('theta.bin')"

epsilon.bin:
	python3 -c "import numpy as np; \
	  dim=8; hid=32; act=64; \
	  sz=dim*hid+hid+hid*act+act; \
	  np.random.normal(0,1,sz).astype('float32').tofile('epsilon.bin')"

# -----------------------------------------------------------------------------
# Test runs (requires GPU)
# -----------------------------------------------------------------------------
test-standalone: gridpack-2d-rl-cuda theta.bin epsilon.bin
	./gridpack-2d-rl-cuda \
	  --gridW=16 --gridH=16 --numBlocks=6 \
	  --theta=theta.bin --epsilon=epsilon.bin \
	  --numThreads=256 --rollouts=64 --mode=0 --seed=12345 \
	  --mlpHiddenDim=32 --actionRegions=8

test-mpi: gridpack-2d-rl-mpi
	mpirun -np 4 ./gridpack-2d-rl-mpi \
	  --gridW=16 --gridH=16 --numBlocks=6 \
	  --numThreads=1024 --rollouts=128 --rounds=5 --sigma=0.01 --lr=0.001 \
	  --mlpHiddenDim=32 --actionRegions=8

# -----------------------------------------------------------------------------
clean:
	rm -f gridpack-2d-rl-cuda gridpack-2d-rl-mpi theta.bin epsilon.bin *.bin

.PHONY: standalone mpi test-standalone test-mpi clean
