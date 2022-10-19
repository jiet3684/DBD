#ifndef __HIP__DEVICE_FUNCTIONS_H__
#define __HIP__DEVICE_FUNCTIONS_H__

#define __DEVICE__ static __host__ __device__ __forceinline__

__DEVICE__ float4 make_float4(float x, float y, float z, float w)
{
  float4 result;
  result.x = x;
  result.y = y;
  result.z = z;
  result.w = w;
  return result;
};

__DEVICE__ float2 make_float2(float x, float y)
{
  float2 result;
  result.x = x;
  result.y = y;
  return result;
};

__DEVICE__ double2 make_double2(double x, double y)
{
  double2 result;
  result.x = x;
  result.y = y;
  return result;
}


#endif
