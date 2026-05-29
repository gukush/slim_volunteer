# MPI + CUDA master-slave kernels

This folder contains MPI master-slave versions of the standalone CUDA kernels.
Goldbach and Pollard reuse the standalone CUDA device code. GridPack follows
the WebGPU strategy shape: rank 0 streams base/perturbed chunks to workers and
integrates returned reward summaries.

## Files

| File | Description |
|------|-------------|
| `mpi_goldbach_verify.cu` | Distributes chunks of even numbers across MPI ranks; each slave runs the Goldbach CUDA kernel on its local GPU. |
| `mpi_pollard_pminus1.cu` | Distributes one number per slave; each slave builds the IO buffer, runs the Pollard p-1 CUDA kernel, and returns the factor. |
| `mpi_gridpack_2d_rl.cu` | Rank 0 sends GridPack base/perturbed chunks to worker ranks; workers evaluate on CUDA and return reward stats. |
| `Makefile` | Builds the MPI CUDA binaries with `nvcc -ccbin mpicxx`. |

## Build

```bash
make
```

If your GPU architecture is different from `sm_75`, edit the `NVCCFLAGS` line in the
`Makefile` (e.g. `-arch=sm_86` for RTX 3060/4060, `-arch=sm_89` for RTX 4070 Ti).

## Running locally (single machine, multiple processes)

```bash
# Goldbach: 4 MPI ranks, each uses its own GPU if available
mpirun --bind-to none -np 4 ./mpi_goldbach_verify --start=4 --end=100000

# Pollard p-1: 4 MPI ranks
mpirun --bind-to none -np 4 ./mpi_pollard_pminus1 --B1=10000 --batch=0x123456789abcdef01

# GridPack: rank 0 is the master, ranks 1..N are CUDA workers
mpirun --bind-to none -np 4 ./mpi_gridpack_2d_rl --numProblems=64 --numThreads=1024 --rollouts=128
```

GridPack also accepts WebGPU-compatible aliases/controls:

```bash
./mpi_gridpack_2d_rl \
  --numProblems=8 \
  --numThreads=1024 \
  --rolloutsPerThread=128 \
  --taskId=a16b4171-69c2-4da3-b4eb-f373dcd47974 \
  --epsilonSeed=12345
```

`--taskId` is converted to the same simple character-sum hash used by the
WebGPU chunker for problem seeds. `--taskIdHash=N` can be used directly.

## Running on N different machines (cluster)

### 1. Create a hostfile

```
192.168.1.10 slots=1
192.168.1.11 slots=1
192.168.1.12 slots=1
```

Use IP addresses or hostnames (hostnames must resolve, e.g. via `/etc/hosts`).

### 2. Passwordless SSH

MPI uses SSH to launch processes on remote nodes. Set up keys once:

```bash
ssh-keygen   # press Enter through defaults
ssh-copy-id user@192.168.1.11
ssh-copy-id user@192.168.1.12
```

### 3. Copy the binary to every node

MPI does **not** copy executables automatically. Put the binary on every node
at the same absolute path, or use an NFS shared folder:

```bash
# Option A: manual scp
scp ./mpi_goldbach_verify user@192.168.1.11:/home/user/myproject/
scp ./mpi_goldbach_verify user@192.168.1.12:/home/user/myproject/

# Option B: NFS shared folder (recommended for clusters)
# /nfs/projects/ is mounted on all nodes
mpirun --hostfile hostfile /nfs/projects/myapp/mpi_goldbach_verify ...
```

### 4. Run

```bash
mpirun --hostfile hostfile --bind-to none ./mpi_goldbach_verify --start=4 --end=100000
```

## Architecture notes

* **Master (rank 0)** reads inputs, chunks work, and `MPI_Send`s data to workers.
* **Workers (rank > 0)** `MPI_Recv` their chunk, copy it to the local GPU,
  launch the CUDA kernel, copy the result back, and `MPI_Send` it to the master.
* GridPack uses dynamic base/perturbed chunk dispatch: `numProblems * 2`
  chunks per round, matching the WebGPU task shape. Adding more worker ranks
  reduces the number of chunks each worker receives, without collective
  `MPI_Bcast`/`MPI_Allreduce`.
* GridPack uses WebGPU-style block generation, `ensureFitsGrid`, 64-thread CUDA
  blocks, and default compile caps matching the generated WGSL defaults
  (`hiddenDim=32`, `actionDim=64`, `MAX_GRID_H=32`).
