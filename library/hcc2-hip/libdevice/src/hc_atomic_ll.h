//
// This file hc_atomic_ll.h was created by /home/grodgers/bin/make_header_from_ll_defines.sh 
// from input file: hc_atomic.ll  
// on Mon May  7 13:22:51 CDT 2018 
//
#ifndef INLINE 
#define INLINE 
#endif
#ifndef _LL_GLOBAL_
#define _LL_GLOBAL_ 
#endif
#ifndef _LL_SHARED_
#define _LL_SHARED_ 
#endif
INLINE unsigned int atomic_exchange_unsigned_global(unsigned int _LL_GLOBAL_ * x, unsigned int y)  ;
INLINE unsigned int atomic_exchange_unsigned_local(unsigned int _LL_SHARED_ * x, unsigned int y)  ;
INLINE unsigned int atomic_exchange_unsigned(unsigned int* x, unsigned int y)  ;
INLINE unsigned int atomic_compare_exchange_unsigned_global(unsigned int _LL_GLOBAL_ * x, unsigned int y, unsigned int z)  ;
INLINE unsigned int atomic_compare_exchange_unsigned_local(unsigned int _LL_SHARED_ * x, unsigned int y, unsigned int z)  ;
INLINE unsigned int atomic_compare_exchange_unsigned(unsigned int* x, unsigned int y, unsigned int z)  ;
INLINE unsigned int atomic_add_unsigned_global(unsigned int _LL_GLOBAL_ * x, unsigned int y)  ;
INLINE unsigned int atomic_add_unsigned_local(unsigned int _LL_SHARED_ * x, unsigned int y)  ;
INLINE unsigned int atomic_add_unsigned(unsigned int* x, unsigned int y)  ;
INLINE unsigned int atomic_sub_unsigned_global(unsigned int _LL_GLOBAL_ * x, unsigned int y)  ;
INLINE unsigned int atomic_sub_unsigned_local(unsigned int _LL_SHARED_ * x, unsigned int y)  ;
INLINE unsigned int atomic_sub_unsigned(unsigned int* x, unsigned int y)  ;
INLINE unsigned int atomic_exchange_int_global(unsigned int _LL_GLOBAL_ * x, unsigned int y)  ;
INLINE unsigned int atomic_exchange_int_local(unsigned int _LL_SHARED_ * x, unsigned int y)  ;
INLINE unsigned int atomic_exchange_int(unsigned int* x, unsigned int y)  ;
INLINE unsigned int atomic_compare_exchange_int_global(unsigned int _LL_GLOBAL_ * x, unsigned int y, unsigned int z)  ;
INLINE unsigned int atomic_compare_exchange_int_local(unsigned int _LL_SHARED_ * x, unsigned int y, unsigned int z)  ;
INLINE unsigned int atomic_compare_exchange_int(unsigned int* x, unsigned int y, unsigned int z)  ;
INLINE unsigned int atomic_add_int_global(unsigned int _LL_GLOBAL_ * x, unsigned int y)  ;
INLINE unsigned int atomic_add_int_local(unsigned int _LL_SHARED_ * x, unsigned int y)  ;
INLINE unsigned int atomic_add_int(unsigned int* x, unsigned int y)  ;
INLINE unsigned int atomic_sub_int_global(unsigned int _LL_GLOBAL_ * x, unsigned int y)  ;
INLINE unsigned int atomic_sub_int_local(unsigned int _LL_SHARED_ * x, unsigned int y)  ;
INLINE unsigned int atomic_sub_int(unsigned int* x, unsigned int y)  ;
INLINE float atomic_exchange_float_global(float _LL_GLOBAL_ * x, float y)  ;
INLINE float atomic_exchange_float_local(float _LL_SHARED_ * x, float y)  ;
INLINE float atomic_exchange_float(float* x, float y)  ;
INLINE float atomic_add_float_global(float _LL_GLOBAL_ * x, float y)  ;
INLINE float atomic_add_float_local(float _LL_SHARED_ * x, float y)  ;
INLINE float atomic_add_float(float* x, float y)  ;
INLINE float atomic_sub_float_global(float _LL_GLOBAL_ * x, float y)  ;
INLINE float atomic_sub_float_local(float _LL_SHARED_ * x, float y)  ;
INLINE float atomic_sub_float(float* x, float y)  ;
INLINE unsigned long long int atomic_exchange_uint64_global(unsigned long long int _LL_GLOBAL_ * x, unsigned long long int y)  ;
INLINE unsigned long long int atomic_exchange_uint64_local(unsigned long long int _LL_SHARED_ * x, unsigned long long int y)  ;
INLINE unsigned long long int atomic_exchange_uint64(unsigned long long int* x, unsigned long long int y)  ;
INLINE unsigned long long int atomic_compare_exchange_uint64_global(unsigned long long int _LL_GLOBAL_ * x, unsigned long long int y, unsigned long long int z)  ;
INLINE unsigned long long int atomic_compare_exchange_uint64_local(unsigned long long int _LL_SHARED_ * x, unsigned long long int y, unsigned long long int z)  ;
INLINE unsigned long long int atomic_compare_exchange_uint64(unsigned long long int* x, unsigned long long int y, unsigned long long int z)  ;
INLINE unsigned long long int atomic_add_uint64_global(unsigned long long int _LL_GLOBAL_ * x, unsigned long long int y)  ;
INLINE unsigned long long int atomic_add_uint64_local(unsigned long long int _LL_SHARED_ * x, unsigned long long int y)  ;
INLINE unsigned long long int atomic_add_uint64(unsigned long long int* x, unsigned long long int y)  ;
INLINE unsigned int atomic_and_unsigned_global(unsigned int _LL_GLOBAL_ * x, unsigned int y)  ;
INLINE unsigned int atomic_or_unsigned_global(unsigned int _LL_GLOBAL_ * x, unsigned int y)  ;
INLINE unsigned int atomic_xor_unsigned_global(unsigned int _LL_GLOBAL_ * x, unsigned int y)  ;
INLINE unsigned int atomic_max_unsigned_global(unsigned int _LL_GLOBAL_ * x, unsigned int y)  ;
INLINE unsigned int atomic_min_unsigned_global(unsigned int _LL_GLOBAL_ * x, unsigned int y)  ;
INLINE unsigned int atomic_and_unsigned_local(unsigned int _LL_SHARED_ * x, unsigned int y)  ;
INLINE unsigned int atomic_or_unsigned_local(unsigned int _LL_SHARED_ * x, unsigned int y)  ;
INLINE unsigned int atomic_xor_unsigned_local(unsigned int _LL_SHARED_ * x, unsigned int y)  ;
INLINE unsigned int atomic_max_unsigned_local(unsigned int _LL_SHARED_ * x, unsigned int y)  ;
INLINE unsigned int atomic_min_unsigned_local(unsigned int _LL_SHARED_ * x, unsigned int y)  ;
INLINE unsigned int atomic_and_unsigned(unsigned int* x, unsigned int y)  ;
INLINE unsigned int atomic_or_unsigned(unsigned int* x, unsigned int y)  ;
INLINE unsigned int atomic_xor_unsigned(unsigned int* x, unsigned int y)  ;
INLINE unsigned int atomic_max_unsigned(unsigned int* x, unsigned int y)  ;
INLINE unsigned int atomic_min_unsigned(unsigned int* x, unsigned int y)  ;
INLINE unsigned int atomic_and_int_global(unsigned int _LL_GLOBAL_ * x, unsigned int y)  ;
INLINE unsigned int atomic_or_int_global(unsigned int _LL_GLOBAL_ * x, unsigned int y)  ;
INLINE unsigned int atomic_xor_int_global(unsigned int _LL_GLOBAL_ * x, unsigned int y)  ;
INLINE unsigned int atomic_max_int_global(unsigned int _LL_GLOBAL_ * x, unsigned int y)  ;
INLINE unsigned int atomic_min_int_global(unsigned int _LL_GLOBAL_ * x, unsigned int y)  ;
INLINE unsigned int atomic_and_int_local(unsigned int _LL_SHARED_ * x, unsigned int y)  ;
INLINE unsigned int atomic_or_int_local(unsigned int _LL_SHARED_ * x, unsigned int y)  ;
INLINE unsigned int atomic_xor_int_local(unsigned int _LL_SHARED_ * x, unsigned int y)  ;
INLINE unsigned int atomic_max_int_local(unsigned int _LL_SHARED_ * x, unsigned int y)  ;
INLINE unsigned int atomic_min_int_local(unsigned int _LL_SHARED_ * x, unsigned int y)  ;
INLINE unsigned int atomic_and_int(unsigned int* x, unsigned int y)  ;
INLINE unsigned int atomic_or_int(unsigned int* x, unsigned int y)  ;
INLINE unsigned int atomic_xor_int(unsigned int* x, unsigned int y)  ;
INLINE unsigned int atomic_max_int(unsigned int* x, unsigned int y)  ;
INLINE unsigned int atomic_min_int(unsigned int* x, unsigned int y)  ;
INLINE unsigned long long int atomic_and_uint64_global(unsigned long long int _LL_GLOBAL_ * x, unsigned long long int y)  ;
INLINE unsigned long long int atomic_or_uint64_global(unsigned long long int _LL_GLOBAL_ * x, unsigned long long int y)  ;
INLINE unsigned long long int atomic_xor_uint64_global(unsigned long long int _LL_GLOBAL_ * x, unsigned long long int y)  ;
INLINE unsigned long long int atomic_max_uint64_global(unsigned long long int _LL_GLOBAL_ * x, unsigned long long int y)  ;
INLINE unsigned long long int atomic_min_uint64_global(unsigned long long int _LL_GLOBAL_ * x, unsigned long long int y)  ;
INLINE unsigned long long int atomic_and_uint64_local(unsigned long long int _LL_SHARED_ * x, unsigned long long int y)  ;
INLINE unsigned long long int atomic_or_uint64_local(unsigned long long int _LL_SHARED_ * x, unsigned long long int y)  ;
INLINE unsigned long long int atomic_xor_uint64_local(unsigned long long int _LL_SHARED_ * x, unsigned long long int y)  ;
INLINE unsigned long long int atomic_max_uint64_local(unsigned long long int _LL_SHARED_ * x, unsigned long long int y)  ;
INLINE unsigned long long int atomic_min_uint64_local(unsigned long long int _LL_SHARED_ * x, unsigned long long int y)  ;
INLINE unsigned long long int atomic_and_uint64(unsigned long long int* x, unsigned long long int y)  ;
INLINE unsigned long long int atomic_or_uint64(unsigned long long int* x, unsigned long long int y)  ;
INLINE unsigned long long int atomic_xor_uint64(unsigned long long int* x, unsigned long long int y)  ;
INLINE unsigned long long int atomic_max_uint64(unsigned long long int* x, unsigned long long int y)  ;
INLINE unsigned long long int atomic_min_uint64(unsigned long long int* x, unsigned long long int y)  ;
INLINE unsigned int atomic_inc_unsigned_global(unsigned int _LL_GLOBAL_ * x)  ;
INLINE unsigned int atomic_dec_unsigned_global(unsigned int _LL_GLOBAL_ * x)  ;
INLINE unsigned int atomic_inc_unsigned_local(unsigned int _LL_SHARED_ * x)  ;
INLINE unsigned int atomic_dec_unsigned_local(unsigned int _LL_SHARED_ * x)  ;
INLINE unsigned int atomic_inc_unsigned(unsigned int* x)  ;
INLINE unsigned int atomic_dec_unsigned(unsigned int* x)  ;
INLINE unsigned int atomic_inc_int_global(unsigned int _LL_GLOBAL_ * x)  ;
INLINE unsigned int atomic_dec_int_global(unsigned int _LL_GLOBAL_ * x)  ;
INLINE unsigned int atomic_inc_int_local(unsigned int _LL_SHARED_ * x)  ;
INLINE unsigned int atomic_dec_int_local(unsigned int _LL_SHARED_ * x)  ;
INLINE unsigned int atomic_inc_int(unsigned int* x)  ;
INLINE unsigned int atomic_dec_int(unsigned int* x)  ;
