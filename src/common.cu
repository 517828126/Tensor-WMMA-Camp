#include "common.h"



template <>
__device__ void data_cast(const float& src, half& dst) {
  dst = __float2half(src);
}

template <>
__device__ void data_cast(const double& src, half& dst) {
  dst = __double2half(src);
}