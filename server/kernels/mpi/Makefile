NVCC = nvcc
NVCCFLAGS = -O2 -arch=sm_75

# Tell nvcc to use mpicc as the host compiler so MPI headers/libs are found.
# Remove -arch=sm_75 or change it (e.g. sm_86) to match your GPU architecture.
MPICC_FLAGS = -ccbin mpicc -Xcompiler -O2

TARGETS = mpi_goldbach_verify mpi_pollard_pminus1

all: $(TARGETS)

mpi_goldbach_verify: mpi_goldbach_verify.cu
	$(NVCC) $(NVCCFLAGS) $(MPICC_FLAGS) $< -o $@ -lmpi

mpi_pollard_pminus1: mpi_pollard_pminus1.cu
	$(NVCC) $(NVCCFLAGS) $(MPICC_FLAGS) $< -o $@ -lmpi

# Convenience run targets (edit -np and args for your cluster)
run-goldbach:
	mpirun --bind-to none -np 4 ./mpi_goldbach_verify --start=4 --end=100000 test 256

run-pollard:
	mpirun --bind-to none -np 4 ./mpi_pollard_pminus1 --B1=10000 --batch=0x123456789abcdef01,0xabcdef0123456789

clean:
	rm -f $(TARGETS)
