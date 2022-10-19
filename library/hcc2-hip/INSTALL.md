# Instructions to build the CLANG HIP compiler

This hcc2-hip repository contains the runtime support for clang hip.
Like cuda, HIP is a kernel definition language that compiles into 
LLVM bytecode for both Radeon and Nvidia GPUs backend compilers.  
The HIP language is triggered by the new "-x hip" clang option.

The clang modifications to support HIP are not all upstream.  
It is our intention to push all this upstream. Till then, you must
build HIP from the clang, llvm, and lld repositories found at 
https://github.com/RadeonOpenCompute. 
Use branch amd-hip for clang and branch amd-commmon for llvm and lld. 

### Build and install HIP compiler and components

1.  Install ROCm Software Stack.

    The instructions on how to install ROCm can be found here:
    <https://github.com/RadeonOpenCompute/ROCm>

    You do not need to install OpenCL. Reboot and make sure that you
    are using the new kernel.  `uname -r` should return something
    like `4.11.0-kfd-compute-rocm-rel-1.6-180`

2.  If you are going to build apps for Nvidia GPUs, then install the 
    NVIDIA CUDA SDK 8.0 (R).

    The CUDA Toolkit 8.0 can be downloaded here:
    <https://developer.nvidia.com/cuda-80-ga2-download-archive>

3.  Download the llvm, clang, and lld source code repositories
    and checkout the appropriate branch. It would help if 
    your local repositories were at $HOME/git/hip
    ```console
    mkdir -p $HOME/git/hip
    cd $HOME/git/hip
    git clone http://github.com/radeonopencompute/clang
    git clone http://github.com/radeonopencompute/llvm
    git clone http://github.com/radeonopencompute/lld
    git clone http://github.com/radeonopencompute/rocm-device-libs
    git clone http://github.com/rocm-developer-tools/hcc2-hip

    cd $HOME/git/hip/clang
    git checkout amd-hip
    cd $HOME/git/hip/llvm
    git checkout amd-common
    cd $HOME/git/hip/lld
    git checkout amd-common
    cd $HOME/git/hip/rocm-device-libs
    git checkout master
    cd $HOME/git/hip/hcc2-hip
    git checkout master
    ```
4.  Build and install the compiler.
    Smart llvm build and install scripts can be found in the bin directory.
    Run these commands.
    ```console
    cd $HOME/git/hip/hcc2-hip/bin
    ./build_amdcommon.sh
    ./build_amdcommon.sh install
    ```
    The build scripts are customizable with environment variables. For example,
    to install the compiler and components in a location other than the default
    "/usr/local/hip" such as "$HOME/install/hip", change the install location 
    with the HIP and SUDO environment variables as follows:
    as follows. 
    ```console
    export HIP=$HOME/install/hip
    export SUDO=noset
    ```
    The command "./build_amdcommon.sh help" will give you more information on the 
    build_amdcommon.sh script. 

5.  Build and install the rocm-device-libs.
    We recommend not building for all gfx processors, just those you need.
    The environment variable GFXLIST controls this. 
    ```console
    export GFXLIST="gfx803 gfx900"
    ./build_libdevice.sh
    ./build_libdevice.sh install
    ```
    Remember to retain the enviroment variable HIP if you changed it.
6.  Build and install the hip device and host runtimes.
    ```console
    ./build_hiprt.sh
    ./build_hiprt.sh install
    ```
    Remember to retain the enviroment variable HIP if you changed it.
7. Test:
    ```console
    cd $HOME/git/hip/hcc2-hip/examples/hip/matrixmul
    make
    make run
    ```
			
