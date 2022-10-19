//===----------------------------------------------------------------------===//
//
// atomics.h  Definitions for hc_atomic.ll functions in rocm device library.
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===/
#ifdef __cplusplus
extern "C" {
#endif
__device__ unsigned atomic_exchange_unsigned(unsigned *addr,
                                             unsigned val);
__device__ unsigned atomic_compare_exchange_unsigned(unsigned *addr,
                                                     unsigned compare,
                                                     unsigned val);
__device__ unsigned atomic_add_unsigned(unsigned *addr,
                                        unsigned val);
__device__ unsigned atomic_sub_unsigned(unsigned *addr,
                                        unsigned val);

__device__ int atomic_exchange_int(int *addr, int val);
__device__ int atomic_compare_exchange_int(int *addr, int compare, int val);
__device__ int atomic_add_int(int *addr, int val);
__device__ int atomic_sub_int(int *addr, int val);

__device__ float atomic_exchange_float(float *addr, float val);
__device__ float atomic_add_float(float *addr, float val);
__device__ float atomic_sub_float(float *addr, float val);

__device__ unsigned long long atomic_exchange_uint64(unsigned long long *addr,
                                                     unsigned long long val);
__device__ unsigned long long
atomic_compare_exchange_uint64(unsigned long long *addr,
                               unsigned long long compare,
                               unsigned long long val);
__device__ unsigned long long atomic_add_uint64(unsigned long long *addr,
                                                unsigned long long val);

__device__ unsigned atomic_and_unsigned(unsigned *addr,
                                        unsigned val);
__device__ unsigned atomic_or_unsigned(unsigned *addr, unsigned val);
__device__ unsigned atomic_xor_unsigned(unsigned *addr, unsigned val);
__device__ unsigned atomic_max_unsigned(unsigned *addr, unsigned val);
__device__ unsigned atomic_min_unsigned(unsigned *addr, unsigned val);

__device__ int atomic_and_int(int *addr, int val);
__device__ int atomic_or_int(int *addr, int val);
__device__ int atomic_xor_int(int *addr, int val);
__device__ int atomic_max_int(int *addr, int val);
__device__ int atomic_min_int(int *addr, int val);

__device__ unsigned long long atomic_and_uint64(unsigned long long *addr,
                                                unsigned long long val);
__device__ unsigned long long atomic_or_uint64(unsigned long long *addr,
                                               unsigned long long val);
__device__ unsigned long long atomic_xor_uint64(unsigned long long *addr,
                                                unsigned long long val);
__device__ unsigned long long atomic_max_uint64(unsigned long long *addr,
                                                unsigned long long val);
__device__ unsigned long long atomic_min_uint64(unsigned long long *addr,
                                                unsigned long long val);

__device__ unsigned atomic_inc_unsigned(unsigned *addr);
__device__ unsigned atomic_dec_unsigned(unsigned *addr);

__device__ int atomic_inc_int(int *addr);
__device__ int atomic_dec_int(int *addr);
#ifdef __cplusplus
}
#endif
