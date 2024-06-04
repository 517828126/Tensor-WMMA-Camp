#include "common.h"

template <>
__device__ void data_cast(const float& src, half& dst) {
  dst = __float2half(src);
}

template <>
__device__ void data_cast(const double& src, half& dst) {
  dst = __double2half(src);
}

template <>
__device__ void data_cast(const float& src, __nv_bfloat16& dst) {
  dst = __float2bfloat16(src);
}

template <>
__device__ void data_cast(const double& src, __nv_bfloat16& dst) {
  dst = __double2bfloat16(src);
}