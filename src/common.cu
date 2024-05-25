#include "common.h"


__global__ void matrix_cast_kernel_float2half(const float* src, half* dst,
                                              size_t row, size_t col,bool transpose ) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  size_t i = idx / col;
  size_t j = idx % col;
  if (i >= row || j >= col) {
    return;
  }
  half target = __float2half(src[i * col + j]);
  if(transpose){
    dst[j * row + i] = target;
  }else{
    dst[i * col + j] = target;
  }
}