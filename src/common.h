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

__device__ void inline data_cast(const float& src, half& dst) {
  dst = __float2half(src);
}
__device__ void inline data_cast(const half& src, float& dst) {
  dst = __half2float(src);
}
__device__ void inline data_cast(const double& src, half& dst) {
  dst = __double2half(src);
}
__device__ void inline data_cast(const float& src, __nv_bfloat16& dst) {
  dst = __float2bfloat16(src);
}
__device__ void inline data_cast(const double& src, __nv_bfloat16& dst) {
  dst = __double2bfloat16(src);
}

template <typename A, typename B>
__global__ void data_preprocess_kernel(B* dst, size_t dpitch, const A* src,
                                       size_t spitch, size_t row, size_t col,
                                       bool transpose) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  size_t i = idx / col;
  size_t j = idx % col;
  if (i >= row || j >= col) {
    return;
  }
  const A* src_row = (const A*)((const char*)src + i * spitch);
  if (transpose) {
    B* dst_row = (B*)((char*)dst + j * dpitch);
    data_cast(src_row[j], dst_row[i]);
  } else {
    B* dst_row = (B*)((char*)dst + i * dpitch);
    data_cast(src_row[j], dst_row[j]);
  }
}

template <typename A>
__global__ void data_preprocess_kernel(A* dst, size_t dpitch, const A* src,
                                       size_t spitch, size_t row, size_t col,
                                       bool transpose) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  size_t i = idx / col;
  size_t j = idx % col;
  if (i >= row || j >= col) {
    return;
  }
  const A* src_row = (const A*)((const char*)src + i * spitch);
  if (transpose) {
    A* dst_row = (A*)((char*)dst + j * dpitch);
    dst_row[i] = src_row[j];
  } else {
    A* dst_row = (A*)((char*)dst + i * dpitch);
    dst_row[j] = src_row[j];
  }
}

template <typename SRC_A, typename SRC_B, typename DEV_A, typename DEV_B,
          typename DEV_C>
void data_preprocess(dim3& grid, const SRC_A* a, const SRC_B* b, const int M,
                     const int N, const int K, DEV_A*& dev_a, DEV_B*& dev_b,
                     DEV_C*& dev_c, int& m, int& n, int& k, int kenerl_m,
                     int kenerl_n, int kenerl_k, bool matrix_b_is_col_major) {
  m = (M + kenerl_m - 1) / kenerl_m;
  n = (N + kenerl_n - 1) / kenerl_n;
  k = (K + kenerl_k - 1) / kenerl_k;
  grid.y = m;
  grid.x = n;
  m *= kenerl_m;
  n *= kenerl_n;
  k *= kenerl_k;
  CUDA_CHECK(cudaMalloc((void**)(&dev_a), m * k * sizeof(DEV_A)));
  CUDA_CHECK(cudaMalloc((void**)(&dev_b), k * n * sizeof(DEV_B)));
  CUDA_CHECK(cudaMalloc((void**)(&dev_c), m * n * sizeof(DEV_C)));
  cudaMemset(dev_a, 0, m * k * sizeof(DEV_A));
  cudaMemset(dev_b, 0, k * n * sizeof(DEV_B));
  cudaMemset(dev_c, 0, m * n * sizeof(DEV_C));
  if (std::is_same_v<SRC_A, DEV_A> && (M == m && K == k)) {
    CUDA_CHECK(
        cudaMemcpy(dev_a, a, m * k * sizeof(SRC_A), cudaMemcpyHostToDevice));
  } else {
    SRC_A* dev_src;
    CUDA_CHECK(cudaMalloc((void**)(&dev_src), M * K * sizeof(SRC_A)));
    CUDA_CHECK(
        cudaMemcpy(dev_src, a, M * K * sizeof(SRC_A), cudaMemcpyHostToDevice));
    int block = ((M * K + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    data_preprocess_kernel<<<block, WARP_SIZE>>>(
        dev_a, k * sizeof(DEV_A), dev_src, K * sizeof(SRC_A), M, K, false);
    cudaFree(dev_src);
  }

  if (std::is_same_v<SRC_B, DEV_B> && (K == k && N == n) &&
      matrix_b_is_col_major) {
    CUDA_CHECK(
        cudaMemcpy(dev_b, b, k * n * sizeof(DEV_B), cudaMemcpyHostToDevice));
  } else {
    SRC_B* dev_src;
    CUDA_CHECK(cudaMalloc((void**)(&dev_src), K * N * sizeof(SRC_B)));
    CUDA_CHECK(
        cudaMemcpy(dev_src, b, K * N * sizeof(SRC_B), cudaMemcpyHostToDevice));
    int block = ((K * N + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    if (matrix_b_is_col_major) {
      data_preprocess_kernel<<<block, WARP_SIZE>>>(
          dev_b, n * sizeof(DEV_B), dev_src, N * sizeof(SRC_B), K, N, false);
    } else {
      data_preprocess_kernel<<<block, WARP_SIZE>>>(
          dev_b, k * sizeof(DEV_B), dev_src, N * sizeof(SRC_B), K, N, true);
    }
    cudaFree(dev_src);
  }
}

#endif