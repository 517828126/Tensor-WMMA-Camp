#ifndef COMMON_H
#define COMMON_H
#include "utils.h"
#include <mma.h>
enum class FP16TensorKenerlType {
  TensorKenerl_16_16_16 = 0,
  TensorKenerl_8_32_16,
  TensorKenerl_32_8_16
};
enum class BF16TensorKenerlType {
  TensorKenerl_16_16_16 = 0,
  TensorKenerl_8_32_16,
  TensorKenerl_32_8_16
};
template <typename A, typename B>
__device__ void data_cast(const A& src, B& dst) {
  dst = static_cast<B>(src);
}

template <>
__device__ void data_cast(const float& src, half& dst);

template <>
__device__ void data_cast(const double& src, half& dst);

template <>
__device__ void data_cast(const float& src, __nv_bfloat16& dst);

template <>
__device__ void data_cast(const double& src, __nv_bfloat16& dst);

template <typename T>
__global__ void matrix_transpose_kernel(T* dst, const T* src, size_t row,
                                        size_t col) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  size_t i = idx / col;
  size_t j = idx % col;
  if (i >= row || j >= col) {
    return;
  }
  dst[j * row + i] = src[i * col + j];
}

template <typename A, typename B>
__global__ void matrix_cast_kernel(B* dst, const A* src, size_t row, size_t col,
                                   bool transpose) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  size_t i = idx / col;
  size_t j = idx % col;
  if (i >= row || j >= col) {
    return;
  }
  if (transpose) {
    data_cast(src[i * col + j], dst[j * row + i]);
  } else {
    data_cast(src[i * col + j], dst[i * col + j]);
  }
}

template <typename A, typename B, typename C>
void data_pre(dim3& grid, int& m, int& n, int& k, int kenerl_m, int kenerl_n,
              int kenerl_k, A** dev_a, B** dev_b, C** dev_c) {
  m = (m + kenerl_m - 1) / kenerl_m;
  n = (n + kenerl_n - 1) / kenerl_n;
  k = (k + kenerl_k - 1) / kenerl_k;
  grid.y = m;
  grid.x = n;
  m *= kenerl_m;
  n *= kenerl_n;
  k *= kenerl_k;
  CUDA_CHECK(cudaMalloc((void**)(dev_a), m * k * sizeof(A)));
  CUDA_CHECK(cudaMalloc((void**)(dev_b), k * n * sizeof(B)));
  CUDA_CHECK(cudaMalloc((void**)(dev_c), m * n * sizeof(C)));
  cudaMemset(dev_a, 0, m * k * sizeof(A));
  cudaMemset(dev_b, 0, k * n * sizeof(B));
  cudaMemset(dev_c, 0, m * n * sizeof(C));
}

template <typename A, typename B, typename T>
void copy_and_cast_data_to_device(const A* a, const B* b, int M, int N, int K,
                                  T* dev_a, T* dev_b, int m, int n, int k,
                                  bool matrix_b_is_col_major) {
  int block_a = (m * k + WARP_SIZE - 1) / WARP_SIZE;
  int block_b = (k * n + WARP_SIZE - 1) / WARP_SIZE;

  A* host_a;
  B* host_b;
  cudaMallocHost((void**)(&host_a), M * K * sizeof(A));
  cudaMallocHost((void**)(&host_b), K * N * sizeof(B));
  memcpy(host_a, a, M * K * sizeof(A));
  memcpy(host_b, b, K * N * sizeof(B));

  A* dev_src_a;
  B* dev_src_b;
  CUDA_CHECK(cudaMalloc((void**)(&dev_src_a), m * k * sizeof(A)));
  CUDA_CHECK(cudaMalloc((void**)(&dev_src_b), k * n * sizeof(B)));
  cudaMemset(dev_src_a, 0, m * k * sizeof(A));
  cudaMemset(dev_src_b, 0, k * n * sizeof(B));
  for (int i = 0; i < M; ++i) {
    CUDA_CHECK(cudaMemcpy(dev_src_a + i * k, host_a + i * K, K * sizeof(A),
                          cudaMemcpyHostToDevice));
  }
  matrix_cast_kernel<<<block_a, WARP_SIZE>>>(dev_a, dev_src_a, m, k, false);
  if (!matrix_b_is_col_major) {
    for (int i = 0; i < K; ++i) {
      CUDA_CHECK(cudaMemcpy(dev_src_b + i * n, host_b + i * N,
                            N * sizeof(float), cudaMemcpyHostToDevice));
    }
    matrix_cast_kernel<<<block_b, WARP_SIZE>>>(dev_b, dev_src_b, k, n, true);
  } else {
    for (int i = 0; i < N; ++i) {
      CUDA_CHECK(cudaMemcpy(dev_src_b + i * k, host_b + i * K,
                            K * sizeof(float), cudaMemcpyHostToDevice));
    }
    matrix_cast_kernel<<<block_b, WARP_SIZE>>>(dev_b, dev_src_b, k, n, false);
  }
  CUDA_CHECK(cudaFree(dev_src_a));
  CUDA_CHECK(cudaFree(dev_src_b));
  CUDA_CHECK(cudaFreeHost(host_a));
  CUDA_CHECK(cudaFreeHost(host_b));
}

template <typename A, typename B>
void copy_data_to_device(const A* a, const B* b, int M, int N, int K, A* dev_a,
                         B* dev_b, int m, int n, int k,
                         bool matrix_b_is_col_major) {
  A* host_a;
  B* host_b;
  cudaMallocHost((void**)(&host_a), M * K * sizeof(A));
  cudaMallocHost((void**)(&host_b), K * N * sizeof(B));
  memcpy(host_a, a, M * K * sizeof(A));
  memcpy(host_b, b, K * N * sizeof(B));
  for (int i = 0; i < M; ++i) {
    CUDA_CHECK(cudaMemcpy(dev_a + i * k, host_a + i * K, K * sizeof(A),
                          cudaMemcpyHostToDevice));
  }
  if (!matrix_b_is_col_major) {
    B* dev_src_b;
    CUDA_CHECK(cudaMalloc((void**)(&dev_src_b), k * n * sizeof(B)));
    cudaMemset(dev_src_b, 0, k * n * sizeof(B));
    for (int i = 0; i < K; ++i) {
      CUDA_CHECK(cudaMemcpy(dev_src_b + i * n, host_b + i * N,
                            N * sizeof(float), cudaMemcpyHostToDevice));
    }
    int block_b = (k * n + WARP_SIZE - 1) / WARP_SIZE;
    matrix_transpose_kernel<<<block_b, WARP_SIZE>>>(dev_b, dev_src_b, k, n);
    CUDA_CHECK(cudaFree(dev_src_b));
  } else {
    for (int i = 0; i < N; ++i) {
      CUDA_CHECK(cudaMemcpy(dev_b + i * k, host_b + i * K, K * sizeof(float),
                            cudaMemcpyHostToDevice));
    }
  }

  CUDA_CHECK(cudaFreeHost(host_a));
  CUDA_CHECK(cudaFreeHost(host_b));
}

#endif