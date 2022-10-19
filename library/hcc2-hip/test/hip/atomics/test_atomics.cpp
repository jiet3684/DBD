// MIT License
//
// Copyright (c) 2017 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person
// obtaining a copy of this software and associated documentation
// files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use, copy,
// modify, merge, publish, distribute, sublicense, and/or sell copies
// of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
// BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
// ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <stdio.h>
#include "hip/hip_runtime.h"

#define N 10

template <typename T>
struct TypeName
{
  static const char* Get()
  {
    return typeid(T).name();
  }
};

template <>
struct TypeName<int>
{
  static const char* Get()
  {
    return "int";
  }
};

template <>
struct TypeName<unsigned>
{
  static const char* Get()
  {
    return "unsigned";
  }
};
template <>
struct TypeName<float>
{
  static const char* Get()
  {
    return "float";
  }
};

template <>
struct TypeName<unsigned long long>
{
  static const char* Get()
  {
    return "unsigned long long";
  }
};

template <typename T>
struct exchangeOp {
  static const char* getName() {
    return "exchangeOp";
  }
  static __device__ T binop(T* addr, T val) {
    return atomicExchange(addr, val);
  }
};

template <typename T>
struct compareExchangeOp {
  static __device__ T ternop(T* addr, T val1, T val2) {
    return atomicCAS(addr, val1, val2);
  }
};

template <typename T>
struct addOp {
  static const char* getName() {
    return "addOp";
  }
  static __device__ T binop(T* addr, T val) {
    return atomicAdd(addr, val);
  }
};

template <typename T>
struct subOp {
  static const char* getName() {
    return "subOp";
  }
  static __device__ T binop(T* addr, T val) {
    return atomicSub(addr, 10);
  }
};

template <typename T>
struct minOp {
  static const char* getName() {
    return "minOp";
  }
  static __device__ T binop(T* addr, T val) {
    return atomicMin(addr, val);
  }
};

template <typename T>
struct maxOp {
  static const char* getName() {
    return "maxOp";
  }
  static __device__ T binop(T* addr, T val) {
    return atomicMax(addr, val);
  }
};

template <typename T>
struct andOp {
  static const char* getName() {
    return "andOp";
  }
  static __device__ T binop(T* addr, T val) {
    return atomicAnd(addr, val);
  }
};

template <typename T>
struct orOp {
  static const char* getName() {
    return "orOp";
  }
  static __device__ T binop(T* addr, T val) {
    return atomicOr(addr, val);
  }
};

template <typename T>
struct xorOp {
  static const char* getName() {
    return "xorOp";
  }
  static __device__ T binop(T* addr, T val) {
    return atomicXor(addr, val);
  }
};

template <typename T>
struct incOp {
  static __device__ T unop(T* addr) {
    return atomicInc(addr);
  }
};

template <typename T>
struct decOp {
  static __device__ T unop(T* addr) {
    return atomicDec(addr);
  }
};

template <typename P, template <typename PP> class T>
__global__ void testUnOp(P* addr) {
  for (int i = 0; i < N; ++ i) {
      T<P>::unop(&addr[i]);
  }
}

template <typename P, template <typename PP> class T>
__global__ void testBinOp(P* addr, P value) {
  for (int i = 0; i < N; ++ i) {
    T<P>::binop(&addr[i], value);
  }
}

template <typename P, template <typename PP> class T>
__global__ void testTernOp(P* addr, P value1, P value2) {
  int i = hipBlockIdx_x;
  if (i<N) {
    T<P>::ternop(addr, value1, value2);
  }
}

template <typename T> void printVector(T *vector)
{
  printf("[");
  bool first = true;
  for (int i = 0; i<N; ++i)
  {
    if (first)
    {
      printf("%d", static_cast<int>(vector[i]));
      first = false;
    }
    else
    {
      printf(", %d", static_cast<int>(vector[i]));
    }
  }
  printf("]");
}

void printHipError(hipError_t error)
{
  printf("Hip Error: %s\n", hipGetErrorString(error));
}

void randomizeVector(int *vector)
{
  for (int i = 0; i < N; ++i)
    vector[i] = rand() % 10;
}

template <typename T> void setVector(T* vector, T value) {
   for (int i = 0; i < N; ++i)
     vector[i] = value;
}

template <typename T> void clearVector(T* vector) {
  setVector<T>(vector, 0);
}

template <typename T> bool checkVector(T* vector, T expectedValue) {
  for (int i = 0; i < N; ++i) {
    if (vector[i] != expectedValue) {
      printf("Error: Found %d instread of %d\n", (int) vector[i], (int)expectedValue);
      return false;
    }
  }
  return true;
}

bool hipCallSuccessful(hipError_t error)
{
  if (error != hipSuccess)
    printHipError(error);
  return error == hipSuccess;
}

bool deviceCanCompute(int deviceID)
{
  bool canCompute = false;
  hipDeviceProp_t deviceProp;
  bool devicePropIsAvailable =
    hipCallSuccessful(hipGetDeviceProperties(&deviceProp, deviceID));
  if (devicePropIsAvailable)
  {
    canCompute = deviceProp.computeMode != hipComputeModeProhibited;
    if (!canCompute)
      printf("Compute mode is prohibited\n");
  }
  return canCompute;
}

bool deviceIsAvailable(int *deviceID)
{
  return hipCallSuccessful(hipGetDevice(deviceID));
}

// We always use device 0
bool haveComputeDevice()
{
  int deviceID = 0;
  return deviceIsAvailable(&deviceID) && deviceCanCompute(deviceID);
}

template <typename T> bool checkResult(T *array, T expected) {
  for (int i = 0; i < N; ++i) {
    if (array[i] != expected) {
      printf("Error: array[%d] = %d, expected %d\n",
             i, (int)array[i], (int)expected);
    }
  }
  return true;
}

template <typename P, template<typename PP> class T>
bool hostTestUnOp() {
  P hostSrcVec[N];
  P hostDstVec[N];

  clearVector<P>(hostSrcVec);
  clearVector<P>(hostDstVec);

  P *deviceSrcVec = NULL;

  printf("  Src: ");
  printVector<P>(hostSrcVec);
  printf("\n  Dst: ");
  printVector<P>(hostDstVec);
  printf("\n");

  bool vectorAAllocated =
    hipCallSuccessful(hipMalloc((void **)&deviceSrcVec, N*sizeof(int)));

  if (vectorAAllocated)
  {
    bool copiedSrcVec =
      hipCallSuccessful(hipMemcpy(deviceSrcVec, hostSrcVec,
                                  N * sizeof(P), hipMemcpyHostToDevice));
    if (copiedSrcVec)
    {
      testUnOp<P, T><<<N,1>>>(deviceSrcVec);
      if (hipCallSuccessful(hipMemcpy(hostDstVec,
                                      deviceSrcVec,
                                      N * sizeof(int),
                                      hipMemcpyDeviceToHost))) {
        printf("Dst: ");
        printVector<P>(hostDstVec);
        printf("\n");
      }
    }
  }

  if (vectorAAllocated)
    hipFree(deviceSrcVec);

  return true;
}
template <typename P, template<typename PP> class T>
bool hostTestBinOp(P vectorValue, P value, P expectedValue) {
  P hostSrcVec[N];
  P hostDstVec[N];

  setVector<P>(hostSrcVec, vectorValue);
  clearVector<P>(hostDstVec);

  P *deviceSrcVec = NULL;

  bool vectorAAllocated =
    hipCallSuccessful(hipMalloc((void **)&deviceSrcVec, N*sizeof(int)));

  if (vectorAAllocated)
  {
    bool copiedSrcVec =
      hipCallSuccessful(hipMemcpy(deviceSrcVec, hostSrcVec,
                                  N * sizeof(P), hipMemcpyHostToDevice));
    if (copiedSrcVec)
    {
      testBinOp<P, T><<<N,1>>>(deviceSrcVec, value);
      if (hipCallSuccessful(hipMemcpy(hostDstVec,
                                      deviceSrcVec,
                                      N * sizeof(P),
                                      hipMemcpyDeviceToHost))) {
        printf("Test: %s<%s>: ", T<P>::getName(), TypeName<P>::Get());
        if (!checkVector<P>(hostDstVec, expectedValue)) {
          printf("Dst: ");
          printVector<P>(hostDstVec);
          printf("\n");

        } else {
          printf("Pass!");
        }
        printf("\n");
      }
    }
  }

  if (vectorAAllocated)
    hipFree(deviceSrcVec);

  return true;
}
template <typename P, template<typename PP> class T>
bool hostTestTernOp() {
  P hostSrcVec[N];
  P hostDstVec[N];

  clearVector<P>(hostSrcVec);
  clearVector<P>(hostDstVec);

  P *deviceSrcVec = NULL;

  printf("  Src: ");
  printVector<P>(hostSrcVec);
  printf("\n  Dst: ");
  printVector<P>(hostDstVec);
  printf("\n");

  bool vectorAAllocated =
    hipCallSuccessful(hipMalloc((void **)&deviceSrcVec, N*sizeof(int)));

  if (vectorAAllocated)
  {
    bool copiedSrcVec =
      hipCallSuccessful(hipMemcpy(deviceSrcVec, hostSrcVec,
                                  N * sizeof(P), hipMemcpyHostToDevice));
    if (copiedSrcVec)
    {
      testTernOp<P, T><<<N,1>>>(deviceSrcVec,
                               static_cast<P>(10),
                               static_cast<P>(10));
      if (hipCallSuccessful(hipMemcpy(hostDstVec,
                                      deviceSrcVec,
                                      N * sizeof(int),
                                      hipMemcpyDeviceToHost))) {
        printf("Dst: ");
        printVector<P>(hostDstVec);
        printf("\n");
      }
    }
  }

  if (vectorAAllocated)
    hipFree(deviceSrcVec);

  return true;
}

int main() {
  if (!haveComputeDevice())
  {
    printf("No compute device available\n");
    return 0;
  }

  hostTestBinOp<unsigned, addOp>(0, 10, 100);
  hostTestBinOp<int, addOp>(0, 10, 100);
  hostTestBinOp<float, addOp>(0, 10.0, 100.0);
  hostTestBinOp<unsigned long long, addOp>(10, 10, 110);

  //  hostTestTernOp<unsigned, compareExchangeOp>();
  //  hostTestTernOp<int, compareExchangeOp>();
  //  hostTestTernOp<unsigned long long, compareExchangeOp>();

  hostTestBinOp<int, subOp>(0, 10, -100);
  hostTestBinOp<unsigned, subOp>(0, 10, -100);

  //  //  hostTestBinOp<int, exchangeOp>();
  //  //  hostTestBinOp<unsigned, exchangeOp>();
  //  //  hostTestBinOp<float, exchangeOp>();
  //  //  hostTestBinOp<unsigned long long, exchangeOp>();

  hostTestBinOp<int, minOp>(0, -10, -10);
  hostTestBinOp<unsigned, minOp>(0, 10, 0);
  hostTestBinOp<unsigned long long, minOp>(0, 10, 0);

  hostTestBinOp<int, maxOp>(0, 10, 10);
  hostTestBinOp<unsigned, maxOp>(0, 10, 10);
  hostTestBinOp<unsigned long long, maxOp>(0, 10, 10);

  hostTestBinOp<int, andOp>(1, 1, 1);
  hostTestBinOp<unsigned, andOp>(0, 1, 0);
  hostTestBinOp<unsigned long long, andOp>(1, 1, 1);

  hostTestBinOp<int, orOp>(0, 1, 1);
  hostTestBinOp<unsigned, orOp>(1, 0, 1);
  hostTestBinOp<unsigned long long, orOp>(1, 1, 1);

  hostTestBinOp<int, xorOp>(0, 1, 0);
  hostTestBinOp<unsigned, xorOp>(0, 1, 0);
  hostTestBinOp<unsigned long long, xorOp>(0, 1, 0);

  //  //  hostTestUnOp<unsigned, incOp>();
  //  //  hostTestUnOp<int, incOp>();
  //  //  hostTestUnOp<unsigned, decOp>();
  //  //  hostTestUnOp<int, decOp>();

  return 0;
}
