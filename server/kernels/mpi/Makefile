NVCC = nvcc
NVCCFLAGS = -O3 -arch=sm_89

# Tell nvcc to use mpicxx as the host compiler so MPI headers/libs are found.
MPICC_FLAGS = -ccbin mpicxx -Xcompiler -O3 -std=c++17

# GridPack-2D RL MLP caps (must be >= runtime values)
MAX_STATE_DIM  = 8
MAX_HIDDEN_DIM = 128
MAX_ACTION_DIM = 256
RL_DEFS = -DMAX_STATE_DIM=$(MAX_STATE_DIM) -DMAX_HIDDEN_DIM=$(MAX_HIDDEN_DIM) -DMAX_ACTION_DIM=$(MAX_ACTION_DIM)

TARGETS = mpi_goldbach_verify mpi_pollard_pminus1 mpi_gridpack_2d_rl

all: $(TARGETS)

mpi_goldbach_verify: mpi_goldbach_verify.cu
	$(NVCC) $(NVCCFLAGS) $(MPICC_FLAGS) $< -o $@ -lmpi

mpi_pollard_pminus1: mpi_pollard_pminus1.cu
	$(NVCC) $(NVCCFLAGS) $(MPICC_FLAGS) $< -o $@ -lmpi

mpi_gridpack_2d_rl: mpi_gridpack_2d_rl.cu
	$(NVCC) $(NVCCFLAGS) $(MPICC_FLAGS) $(RL_DEFS) $< -o $@ -lmpi

# Convenience run targets (edit -np and args for your cluster)
run-goldbach:
	mpirun --bind-to none -np 4 ./mpi_goldbach_verify --start=4 --end=100000 test 256

run-pollard:
	mpirun --bind-to none -np 4 ./mpi_pollard_pminus1 --B1=10000 --batch=0x123456789abcdef01,0xabcdef0123456789

run-gridpack:
	mpirun --bind-to none -np 4 ./mpi_gridpack_2d_rl \
	  --gridW=16 --gridH=16 --numBlocks=6 \
	  --numThreads=1024 --rollouts=128 --rounds=1 --sigma=0.01 --lr=0.001 \
	  --mlpHiddenDim=32 --actionRegions=8

clean:
	rm -f $(TARGETS)
