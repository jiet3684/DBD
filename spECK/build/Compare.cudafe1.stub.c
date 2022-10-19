#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_CUBIN_HANDLE_STORAGE__ static
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "Compare.fatbin.c"
static void __device_stub__Z9d_compareIfEviiPKjS1_PKT_S1_S1_S4_bdPj(int, int, const uint32_t *__restrict__, const uint32_t *__restrict__, const float *__restrict__, const uint32_t *__restrict__, const uint32_t *__restrict__, const float *__restrict__, bool, double, uint32_t *);
static void __device_stub__Z9d_compareIdEviiPKjS1_PKT_S1_S1_S4_bdPj(int, int, const uint32_t *__restrict__, const uint32_t *__restrict__, const double *__restrict__, const uint32_t *__restrict__, const uint32_t *__restrict__, const double *__restrict__, bool, double, uint32_t *);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void) __attribute__((__constructor__));
static void __device_stub__Z9d_compareIfEviiPKjS1_PKT_S1_S1_S4_bdPj(int __par0, int __par1, const uint32_t *__restrict__ __par2, const uint32_t *__restrict__ __par3, const float *__restrict__ __par4, const uint32_t *__restrict__ __par5, const uint32_t *__restrict__ __par6, const float *__restrict__ __par7, bool __par8, double __par9, uint32_t *__par10){ const uint32_t *__T0;
 const uint32_t *__T1;
 const float *__T2;
 const uint32_t *__T3;
 const uint32_t *__T4;
 const float *__T5;
__cudaLaunchPrologue(11);__cudaSetupArgSimple(__par0, 0UL);__cudaSetupArgSimple(__par1, 4UL);__T0 = __par2;__cudaSetupArgSimple(__T0, 8UL);__T1 = __par3;__cudaSetupArgSimple(__T1, 16UL);__T2 = __par4;__cudaSetupArgSimple(__T2, 24UL);__T3 = __par5;__cudaSetupArgSimple(__T3, 32UL);__T4 = __par6;__cudaSetupArgSimple(__T4, 40UL);__T5 = __par7;__cudaSetupArgSimple(__T5, 48UL);__cudaSetupArgSimple(__par8, 56UL);__cudaSetupArgSimple(__par9, 64UL);__cudaSetupArgSimple(__par10, 72UL);__cudaLaunch(((char *)((void ( *)(int, int, const uint32_t *__restrict__, const uint32_t *__restrict__, const float *__restrict__, const uint32_t *__restrict__, const uint32_t *__restrict__, const float *__restrict__, bool, double, uint32_t *))d_compare<float> )));}
template<> __specialization_static void __wrapper__device_stub_d_compare<float>( int &__cuda_0,int &__cuda_1,const ::uint32_t *__restrict__ &__cuda_2,const ::uint32_t *__restrict__ &__cuda_3,const float *__restrict__ &__cuda_4,const ::uint32_t *__restrict__ &__cuda_5,const ::uint32_t *__restrict__ &__cuda_6,const float *__restrict__ &__cuda_7,bool &__cuda_8,double &__cuda_9,::uint32_t *&__cuda_10){__device_stub__Z9d_compareIfEviiPKjS1_PKT_S1_S1_S4_bdPj( (int &)__cuda_0,(int &)__cuda_1,(const ::uint32_t *&)__cuda_2,(const ::uint32_t *&)__cuda_3,(const float *&)__cuda_4,(const ::uint32_t *&)__cuda_5,(const ::uint32_t *&)__cuda_6,(const float *&)__cuda_7,(bool &)__cuda_8,(double &)__cuda_9,(::uint32_t *&)__cuda_10);}
static void __device_stub__Z9d_compareIdEviiPKjS1_PKT_S1_S1_S4_bdPj(int __par0, int __par1, const uint32_t *__restrict__ __par2, const uint32_t *__restrict__ __par3, const double *__restrict__ __par4, const uint32_t *__restrict__ __par5, const uint32_t *__restrict__ __par6, const double *__restrict__ __par7, bool __par8, double __par9, uint32_t *__par10){ const uint32_t *__T6;
 const uint32_t *__T7;
 const double *__T8;
 const uint32_t *__T9;
 const uint32_t *__T10;
 const double *__T11;
__cudaLaunchPrologue(11);__cudaSetupArgSimple(__par0, 0UL);__cudaSetupArgSimple(__par1, 4UL);__T6 = __par2;__cudaSetupArgSimple(__T6, 8UL);__T7 = __par3;__cudaSetupArgSimple(__T7, 16UL);__T8 = __par4;__cudaSetupArgSimple(__T8, 24UL);__T9 = __par5;__cudaSetupArgSimple(__T9, 32UL);__T10 = __par6;__cudaSetupArgSimple(__T10, 40UL);__T11 = __par7;__cudaSetupArgSimple(__T11, 48UL);__cudaSetupArgSimple(__par8, 56UL);__cudaSetupArgSimple(__par9, 64UL);__cudaSetupArgSimple(__par10, 72UL);__cudaLaunch(((char *)((void ( *)(int, int, const uint32_t *__restrict__, const uint32_t *__restrict__, const double *__restrict__, const uint32_t *__restrict__, const uint32_t *__restrict__, const double *__restrict__, bool, double, uint32_t *))d_compare<double> )));}
template<> __specialization_static void __wrapper__device_stub_d_compare<double>( int &__cuda_0,int &__cuda_1,const ::uint32_t *__restrict__ &__cuda_2,const ::uint32_t *__restrict__ &__cuda_3,const double *__restrict__ &__cuda_4,const ::uint32_t *__restrict__ &__cuda_5,const ::uint32_t *__restrict__ &__cuda_6,const double *__restrict__ &__cuda_7,bool &__cuda_8,double &__cuda_9,::uint32_t *&__cuda_10){__device_stub__Z9d_compareIdEviiPKjS1_PKT_S1_S1_S4_bdPj( (int &)__cuda_0,(int &)__cuda_1,(const ::uint32_t *&)__cuda_2,(const ::uint32_t *&)__cuda_3,(const double *&)__cuda_4,(const ::uint32_t *&)__cuda_5,(const ::uint32_t *&)__cuda_6,(const double *&)__cuda_7,(bool &)__cuda_8,(double &)__cuda_9,(::uint32_t *&)__cuda_10);}
static void __nv_cudaEntityRegisterCallback(void **__T15){__nv_dummy_param_ref(__T15);__nv_save_fatbinhandle_for_managed_rt(__T15);__cudaRegisterEntry(__T15, ((void ( *)(int, int, const uint32_t *__restrict__, const uint32_t *__restrict__, const double *__restrict__, const uint32_t *__restrict__, const uint32_t *__restrict__, const double *__restrict__, bool, double, uint32_t *))d_compare<double> ), _Z9d_compareIdEviiPKjS1_PKT_S1_S1_S4_bdPj, (-1));__cudaRegisterEntry(__T15, ((void ( *)(int, int, const uint32_t *__restrict__, const uint32_t *__restrict__, const float *__restrict__, const uint32_t *__restrict__, const uint32_t *__restrict__, const float *__restrict__, bool, double, uint32_t *))d_compare<float> ), _Z9d_compareIfEviiPKjS1_PKT_S1_S1_S4_bdPj, (-1));}
static void __sti____cudaRegisterAll(void){__cudaRegisterBinary(__nv_cudaEntityRegisterCallback);}

#pragma GCC diagnostic pop
