# Dynamic Block Distributor

## Compilation

We provide some object files (.o), and cuda source code files (.cu) since .cu files need to be compiled with appropriate compute capability.
All object files are included in dbd/for_small_datasets/obj/ or dbd/for_large_datasets/obj directory, and cuda source codes are included in dbd/for_small_datasets/src or dbd/for_large_datasets/src directory.

dbd/for_small_datasets is a Dynamic Block Distributor framework when blocksize is 4096.
Small datasets are evaluated with this small blocksize since they do not require large memory space.

Also, dbd/for_large_datadsets is a Dynamic Block Distributor framework when blocksize is 512.
Large datasets are evaluated with this blocksize.


In base directory, such as dbd/for_small_datasets or dbd/for_large_datasets, you need to modify the given Makefile based on your 1) base directory, 2) Compute Capability (compute_xx, sm_xx).

After the process, you can generate execution files with "$ make".

This command generates three binary files bin/compute, bin/d2h, and bin/SSpMM, for evaluation of kernel, kernel + D2H, kernel + D2H + I/O, respectively.

"$ make debug" generates bin/debug and you can find out I/O binding ratio with it.

DBD can be compiled with any version of CUDA.

## Baseline
We used Intel MKL 2022.1 [1], NVIDIA cuSPARSE v11.7 [2], bhSPARSE [3], CUSP [4], and spECK [5] for baselines.
Every baseline source codes are also provided.
bhSPARSE and CUSP need some additional libraries and they are included in library directory.
You need to modify the Makefile with appropriate "LIBRARY_PATH".

You need Intel MKL for evaluation.
[Intel MKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html)

After the installation, move to mkl directory and run "$ source /opt/intel/oneapi/setvars.sh" to use MKL libary.

You can get spECK in [spECK](https://github.com/GPUPeople/spECK), but it need to be modified for CUDA 11.7. So we included modified version in spECK directory.

## Environments
OS: Ubuntu 20.04

CPU: Intel i7-11700k

GPU: NVIDA RTX 3090 (Capability 8.6, Ampere Architecture)

CUDA Version: 11.7

Memory: 32 GiB

SSDs: Samsung NVMe PM981


# Usage

Running ./bin/SSpMM without any arguments prints its instructions.

You can run DBD with input file name and DRAM size (GiB). DRAM size effects the size of result buffer in host DRAM.

<1> Executable File Name, Input File 1  (Performs A * A with CPU DRAM size is 28 GiB, GPU DRAM size is 22 GiB)

<2> Executable File Name, Input File 1, Input File 2 (Performs A * B with CPU DRAM size is 28 GiB, GPU DRAM size is 22 GiB)

<3> Executable File Name, CPU DRAM Size, GPU DRAM Size, Input File 1 (Performs A * A with passed CPU&GPU DRAM Size)

<4> Executable File Name, CPU DRAM Size, GPU DRAM Size, Input File 1, Input File 2 (Performs A * B with passed CPU&GPU DRAM Size)

Ex) ./bin/SSpMM 24 16 data

All datasets can be found in [SuiteSparse Matrix Collection](https://sparse.tamu.edu/)

MKL and cuSPARSE read .csr files made by DBD framework. Therefore, please run DBD first, and then run mkl or cusparse for the same input data.

# Result
Every execution Time includes kernel execution, device-to-host transfer (if needed), and file I/O.

Note that execution time of Dynamic Block Distribution also includes preprocessing overheads (block segmentation, categorization).

bin/compute: Kernel

bin/d2h: Kernel + D2H

bin/SSpMM: Kernel + D2H + File I/O

## References
[1] "Intel Math Kernel Library", https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html

[2] "NVIDIA cuSPARSE v11.7", https://docs.nvidia.com/cuda/cusparse/index.html

[3] Weifeng   Liu   and   Brian   Vinter.A   framework   for   general   sparsematrix–matrix  multiplication  on  gpus  and  heterogeneous  processors.Journal of Parallel and Distributed Computing, 85:47–61, 2015. IPDPS2014 Selected Papers on Numerical and Combinatorial Algorithms

[4] N  Bell  and  M  Garland.   Cusp:  Generic  parallel  algorithms  for  sparsematrix and graph computations, version 0.5.1, 2015.

[5] Mathias Parger, Martin Winter, Daniel Mlakar, and Markus Steinberger.Speck:  Accelerating  gpu  sparse  matrix-matrix  multiplication  throughlightweight analysis. InProceedings of the 25th ACM SIGPLAN Sympo-sium on Principles and Practice of Parallel Programming, PPoPP ’20,page 362–375, New York, NY, USA, 2020. Association for ComputingMachinery.


* We will provide the full source code soon, when some confidential issues are resolved.
