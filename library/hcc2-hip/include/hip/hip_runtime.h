/*
Copyright (c) 2015 - present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

//! HIP = Heterogeneous-compute Interface for Portability
//!
//! Define a extremely thin runtime layer that allows source code to be compiled unmodified
//! through either AMD HCC or NVCC.   Key features tend to be in the spirit
//! and terminology of CUDA, but with a portable path to other accelerators as well:
//
//! Both paths support rich C++ features including classes, templates, lambdas, etc.
//! Runtime API is C
//! Memory management is based on pure pointers and resembles malloc/free/copy.
//
//! hip_runtime.h     : includes everything in hip_api.h, plus math builtins and kernel launch macros.
//! hip_runtime_api.h : Defines HIP API.  This is a C header file and does not use any C++ features.

#ifndef HIP_INCLUDE_HIP_HIP_RUNTIME_H
#define HIP_INCLUDE_HIP_HIP_RUNTIME_H

#if defined __HCC__
#define __HIP_PLATFORM_HCC__
// For nvidia targets, the clang cuda wrappers will turn on __CUDACC__
// for both the host and device passes. __CUDACC__ will not be set
// for amdgcn targets. 
#elif defined(__HIP__) && !defined(__CUDACC__)
#define __HIP_PLATFORM_CLANG__
#else
// For "-x hip" on nvidia targets, or or any non-clang compiler, 
// assume compile-time convert hip to cuda
#define __HIP_PLATFORM_NVCC__
#endif 

#if defined __HIP_PLATFORM_CLANG__
#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
#define __shared__ __attribute__((shared))
#define __managed__ __attribute__((managed))
#define __host__ __attribute__((host))
#define __constant__ __attribute__((constant))
#define __noinline__ __attribute__((noinline))

#include <cstdlib>
extern "C" {
// We need these declarations and wrappers for device-side
// malloc/free/printf calls to work without relying on
// -fcuda-disable-target-call-checks option.
__device__ int vprintf(const char *, const char *);
__device__ void free(void *) __attribute((nothrow));
__device__ void *malloc(size_t) __attribute((nothrow)) __attribute__((malloc));
__device__ void __assertfail(const char *__message, const char *__file,
                             unsigned __line, const char *__function,
                             size_t __charSize) __attribute__((noreturn));
// In order for standard assert() macro on linux to work we need to
// provide device-side __assert_fail()
__device__ static inline void __assert_fail(const char *__message,
                                            const char *__file, unsigned __line,
                                            const char *__function) {
  __assertfail(__message, __file, __line, __function, sizeof(char));
}

// Clang will convert printf into vprintf, but we still need
// device-side declaration for it.
__device__ int printf(const char *, ...);
} // extern "C"

#endif // __HIP_PLATFORM_CLANG__

// Some standard header files, these are included by hc.hpp and so want to make them avail on both
// paths to provide a consistent include env and avoid "missing symbol" errors that only appears
// on NVCC path:
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#if __cplusplus > 199711L
#include <thread>
#endif

#include <hip/hip_common.h>

#if defined(__HIP_PLATFORM_HCC__) && !defined (__HIP_PLATFORM_NVCC__) && !defined (__HIP_PLATFORM_CLANG__)
#include <hip/hcc_detail/hip_runtime.h>
#elif defined(__HIP_PLATFORM_NVCC__) && !defined (__HIP_PLATFORM_HCC__) && !defined (__HIP_PLATFORM_CLANG__)
#include <hip/nvcc_detail/hip_runtime.h>
#elif defined(__HIP_PLATFORM_CLANG__) && !defined (__HIP_PLATFORM_HCC__) && !defined (__HIP_PLATFORM_NVCC__)
#include <hip/clang_detail/hip_runtime.h>
#else
#error("Define one: __HIP_PLATFORM_HCC__ __HIP_PLATFORM_NVCC__ or __HIP_PLATFORM_CLANG__");
#endif

#include <hip/hip_runtime_api.h>
#include <hip/hip_vector_types.h>

#endif
