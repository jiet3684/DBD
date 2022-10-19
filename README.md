# DBD

## Compile
Because of funding issue, we cannot provide the full code yet. Sorry for that.

We will provide some object file (.o), and cuda source code (.cu) since .cu file need to be compiled with appropriate compute capability.
All object files present in icde/small/obj/ or icde/large/obj directory.
And two cuda source codes present in icde/small/src or icde/large/src directory.

icde/small is Dynamic Block Distributor framework with blocksize is 4096.
Small datasets in table are evaluated with this blocksize since they do not need large memory space.

Hence, icde/large is Dynamic Block Distributor framework with blocksize is 512.
Large datasets in table are evaluated with this blocksize.


In base directory, such as icde/small or icde/large, you need to modify Makefile for 1) base directory, 2) Compute Capability (compute_xx, sm_xx).

After that, you can generate execution file with "$ make".

This command generates three binary files bin/compute, bin/d2h, bin/SSpMM each for evaluation of kernel, kernel + D2H, kernel + D2H + I/O.

"$ make debug" generates bin/debug and you can find out I/O binding ratio with it.

## Baseline
We used Intel MKL 2022.1 [1], NVIDIA cuSPARSE v11.7 [2], bhSPARSE [3], CUSP [4], spECK [5] for baselines.
Every baseline source codes except spECK also presents in here.
bhSPARSE and CUSP need some additional libraries and they are included in library directory.
You need to modify Makefile, which presents in bhsparse, cusp directory, with appropriate "LIBRARY_PATH"

You need Intel MKL for evaluation.
[Intel MKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html)

After installation, move to mkl directory and run "source /opt/intel/oneapi/setvars.sh" to use MKL libary.

You can get spECK in [spECK](https://github.com/GPUPeople/spECK), but it need to be modified for CUDA 11.7. So we included modified version in spECK directory.

## Environments
OS: Ubuntu 20.04

CPU: Intel i7-11700k

GPU: NVIDA RTX 3090 (Capability 8.6, Ampere Architecture)

CUDA Version: 11.7

Memory: 32 GiB

SSDs: Samsung NVMe PM981


# Usage

Just running ./bin/SSpMM prints usage.

You can run with input file name, DRAM size (GiB). DRAM size effects to size of result buffer in host DRAM.

<1> Executable File Name, Input File 1  (Performs A * A with CPU DRAM size is 28 GiB, GPU DRAM size is 22 GiB)

<2> Executable File Name, Input File 1, Input File 2 (Performs A * B with CPU DRAM size is 28 GiB, GPU DRAM size is 22 GiB)

<3> Executable File Name, CPU DRAM Size, GPU DRAM Size, Input File 1 (Performs A * A with passed CPU&GPU DRAM Size)

<4> Executable File Name, CPU DRAM Size, GPU DRAM Size, Input File 1, Input File 2 (Performs A * B with passed CPU&GPU DRAM Size)

Ex) ./bin/SSpMM 24 16 data

All datasets used can be found in [SuiteSparse Matrix Collection](https://sparse.tamu.edu/)

MKL and cuSPARSE reads .csr file made by DBD framework. So run DBD first, and then run mkl or cusparse for same input data.

# Result
Every execution Time includes kernel execution, device-to-host transfer (if needed), file I/O.

And execution time of Dynamic Block Distribution also includes preprocessing overhead (block segmentation, categorization).

## References
[1] "Intel Math Kernel Library", https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html

[2] "NVIDIA cuSPARSE v11.7", https://docs.nvidia.com/cuda/cusparse/index.html

[3] Weifeng   Liu   and   Brian   Vinter.A   framework   for   general   sparsematrix–matrix  multiplication  on  gpus  and  heterogeneous  processors.Journal of Parallel and Distributed Computing, 85:47–61, 2015. IPDPS2014 Selected Papers on Numerical and Combinatorial Algorithms

[4] N  Bell  and  M  Garland.   Cusp:  Generic  parallel  algorithms  for  sparsematrix and graph computations, version 0.5.1, 2015.

[5] Mathias Parger, Martin Winter, Daniel Mlakar, and Markus Steinberger.Speck:  Accelerating  gpu  sparse  matrix-matrix  multiplication  throughlightweight analysis. InProceedings of the 25th ACM SIGPLAN Sympo-sium on Principles and Practice of Parallel Programming, PPoPP ’20,page 362–375, New York, NY, USA, 2020. Association for ComputingMachinery.
