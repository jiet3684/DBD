# MIT License
#
# Copyright (c) 2017 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use, copy,
# modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Try to detect in the system several dependencies required by the different
# components of libhiprt. These are the dependencies we have:
#
# libelf : required by some targets to handle the ELF files at runtime.
# libffi : required to launch target kernels given function and argument
#          pointers.
# CUDA : required to control offloading to NVIDIA GPUs.

include (FindPackageHandleStandardArgs)

find_package(PkgConfig)

################################################################################
# Looking for CUDA...
################################################################################
find_package(CUDA QUIET)

set(LIBHIPRT_DEP_CUDA_FOUND ${CUDA_FOUND})
set(LIBHIPRT_DEP_CUDA_LIBRARIES ${CUDA_LIBRARIES})
set(LIBHIPRT_DEP_CUDA_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS})

mark_as_advanced(
  LIBHIPRT_DEP_CUDA_FOUND
  LIBHIPRT_DEP_CUDA_INCLUDE_DIRS
  LIBHIPRT_DEP_CUDA_LIBRARIES)


################################################################################
# Looking for ROCM...
################################################################################
pkg_check_modules(LIBHIPRT_SEARCH_LIBHSA QUIET libhsa-runtime64)

find_path (
  LIBHIPRT_DEP_LIBHSA_INCLUDE_DIRS
  NAMES
  hsa.h
  PATHS
  $ENV{HSA_RUNTIME_PATH}/include
  /opt/rocm/include/hsa
  /usr/local/include
  )

find_path (
  LIBHIPRT_DEP_LIBHSA_LIBRARIES_DIRS
  NAMES
  libhsa-runtime64.so
  PATHS
  $ENV{HSA_RUNTIME_PATH}/lib
  /opt/rocm/lib/
  /usr/local/lib
  )


################################################################################
# Looking for AMDGCN devce compiler
################################################################################

find_package(LLVM 7 CONFIG
  PATHS
  $ENV{HIP}
  /usr/local/hip
  $ENV{HCC2}
  /opt/rocm/hcc2
  )

if (LLVM_DIR)
  message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}. Configure: ${LLVM_DIR}/LLVMConfig.cmake")
endif()


