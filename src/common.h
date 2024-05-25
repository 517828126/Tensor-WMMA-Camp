#ifndef COMMON_H
#define COMMON_H
#include "utils.h"
#include <mma.h>
template <typename T>
__global__ void matrix_transpose_kernel(const T* src, T* dst, size_t row,
                                        size_t col) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  size_t i = idx / col;
  size_t j = idx % col;
  if (i >= row || j >= col) {
    return;
  }
  dst[j * row + i] = src[i * col + j];
}

template <typename T>
int matrix_transpose(const T* src, T* dst, size_t row, size_t col) {
  T *d_src, d_dst;
  cudaMalloc((void**)(&d_src), row * col * sizeof(T));
  cudaMalloc((void**)(&d_dst), col * row * sizeof(T));
  cudaMemcpy(d_src, src, row * col * sizeof(T), cudaMemcpyHostToDevice);
  size_t block_size = (row * col + 31) / 32;
  matrix_transpose_kernel<<<block_size, 32>>>(d_src, d_dst, row, col);
}

__global__ void matrix_cast_kernel_float2half(const float* src, half* dst,
                                              size_t row, size_t col,
                                              bool transpose = false);

#endif