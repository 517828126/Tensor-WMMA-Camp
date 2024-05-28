#include "wmma_function.h"
using namespace nvcuda;
__global__ void wmma_fp16_kernel_16_16_16(const half *a, const half *b, int M,
                                          int N, float *c, int K) {
  size_t row = 16 * blockIdx.y;
  size_t col = 16 * blockIdx.x;

  if (row >= M || col >= N) {
    return;
  }

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
  wmma::fill_fragment(c_frag, 0.0f);
  size_t loop_n = (K + 15) / 16;
#pragma unroll
  for (size_t i = 0; i < loop_n; ++i) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::load_matrix_sync(a_frag, a + row * K + i * 16, K);
    wmma::load_matrix_sync(b_frag, b + col * K + i * 16, K);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }
  wmma::store_matrix_sync(c + row * N + col, c_frag, N, wmma::mem_row_major);
}
__global__ void wmma_fp16_kernel_8_32_16(const half *a, const half *b, int M,
                                         int N, float *c, int K) {
  size_t row = 8 * blockIdx.y;
  size_t col = 32 * blockIdx.x;

  if (row >= M || col >= N) {
    return;
  }

  wmma::fragment<wmma::accumulator, 8, 32, 16, float> c_frag;
  wmma::fill_fragment(c_frag, 0.0f);
  size_t loop_n = (K + 15) / 16;
#pragma unroll
  for (size_t i = 0; i < loop_n; ++i) {
    wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::col_major> b_frag;
    wmma::load_matrix_sync(a_frag, a + row * K + i * 16, K);
    wmma::load_matrix_sync(b_frag, b + col * K + i * 16, K);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }
  wmma::store_matrix_sync(c + row * N + col, c_frag, N, wmma::mem_row_major);
}

__global__ void wmma_fp16_kernel_32_8_16(const half *a, const half *b, int M,
                                         int N, float *c, int K) {
  size_t row = 32 * blockIdx.y;
  size_t col = 8 * blockIdx.x;

  if (row >= M || col >= N) {
    return;
  }

  wmma::fragment<wmma::accumulator, 32, 8, 16, float> c_frag;
  wmma::fill_fragment(c_frag, 0.0f);
  size_t loop_n = (K + 15) / 16;
#pragma unroll
  for (size_t i = 0; i < loop_n; ++i) {
    wmma::fragment<wmma::matrix_a, 32, 8, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 32, 8, 16, half, wmma::col_major> b_frag;
    wmma::load_matrix_sync(a_frag, a + row * K + i * 16, K);
    wmma::load_matrix_sync(b_frag, b + col * K + i * 16, K);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }
  wmma::store_matrix_sync(c + row * N + col, c_frag, N, wmma::mem_row_major);
}

int wmma_fp16(const float *a, const float *b, int M, int N, float *c, int K,
              bool matrix_b_is_col_major, FP16TensorKenerlType kenerl_type) {
  if (M < 0 || N < 0 || K < 0) {
    return -1;
  }
  int kenerl_m = 0, kenerl_n = 0, kenerl_k = 0;
  switch (kenerl_type) {
    case FP16TensorKenerlType::TensorKenerl_16_16_16:
      kenerl_m = 16;
      kenerl_n = 16;
      kenerl_k = 16;
      break;
    case FP16TensorKenerlType::TensorKenerl_8_32_16:
      kenerl_m = 8;
      kenerl_n = 32;
      kenerl_k = 16;
      break;
    case FP16TensorKenerlType::TensorKenerl_32_8_16:
      kenerl_m = 32;
      kenerl_n = 8;
      kenerl_k = 16;
      break;
  }
  if (kenerl_m == 0 || kenerl_n == 0 || kenerl_k == 0) {
    return -1;
  }
  dim3 grid;
  int m = M;
  int n = N;
  int k = K;
  half *dev_a;
  half *dev_b;
  float *dev_c;
  data_pre(grid, m, n, k, kenerl_m, kenerl_n, kenerl_k, &dev_a, &dev_b, &dev_c);

  copy_and_cast_data_to_device(a, b, M, N, K, dev_a, dev_b, m, n, k,
                               matrix_b_is_col_major);
  switch (kenerl_type) {
    case FP16TensorKenerlType::TensorKenerl_16_16_16:
      wmma_fp16_kernel_16_16_16<<<grid, WARP_SIZE>>>(dev_a, dev_b, m, n, dev_c,
                                                     k);
      break;
    case FP16TensorKenerlType::TensorKenerl_8_32_16:
      wmma_fp16_kernel_8_32_16<<<grid, WARP_SIZE>>>(dev_a, dev_b, m, n, dev_c,
                                                    k);
      break;
    case FP16TensorKenerlType::TensorKenerl_32_8_16:
      wmma_fp16_kernel_32_8_16<<<grid, WARP_SIZE>>>(dev_a, dev_b, m, n, dev_c,
                                                    k);
      break;
  }
  for (int i = 0; i < M; ++i) {
    CUDA_CHECK(cudaMemcpy(c + i * N, dev_c + i * n, N * sizeof(float),
                          cudaMemcpyDeviceToHost));
  }
  CUDA_CHECK(cudaFree(dev_a));
  CUDA_CHECK(cudaFree(dev_b));
  CUDA_CHECK(cudaFree(dev_c));
  return -1;
}

int wmma_fp16(const half *a, const half *b, int M, int N, float *c, int K,
              bool matrix_b_is_col_major, FP16TensorKenerlType kenerl_type) {
  if (M < 0 || N < 0 || K < 0) {
    return -1;
  }
  int kenerl_m = 0, kenerl_n = 0, kenerl_k = 0;
  switch (kenerl_type) {
    case FP16TensorKenerlType::TensorKenerl_16_16_16:
      kenerl_m = 16;
      kenerl_n = 16;
      kenerl_k = 16;
      break;
    case FP16TensorKenerlType::TensorKenerl_8_32_16:
      kenerl_m = 8;
      kenerl_n = 32;
      kenerl_k = 16;
      break;
    case FP16TensorKenerlType::TensorKenerl_32_8_16:
      kenerl_m = 32;
      kenerl_n = 8;
      kenerl_k = 16;
      break;
  }
  if (kenerl_m == 0 || kenerl_n == 0 || kenerl_k == 0) {
    return -1;
  }
  dim3 grid;
  int m = M;
  int n = N;
  int k = K;
  half *dev_a;
  half *dev_b;
  float *dev_c;
  data_pre(grid, m, n, k, kenerl_m, kenerl_n, kenerl_k, &dev_a, &dev_b, &dev_c);
  copy_data_to_device(a, b, M, N, K, dev_a, dev_b, m, n, k,
                      matrix_b_is_col_major);
  switch (kenerl_type) {
    case FP16TensorKenerlType::TensorKenerl_16_16_16:
      wmma_fp16_kernel_16_16_16<<<grid, WARP_SIZE>>>(dev_a, dev_b, m, n, dev_c,
                                                     k);
      break;
    case FP16TensorKenerlType::TensorKenerl_8_32_16:
      wmma_fp16_kernel_8_32_16<<<grid, WARP_SIZE>>>(dev_a, dev_b, m, n, dev_c,
                                                    k);
      break;
    case FP16TensorKenerlType::TensorKenerl_32_8_16:
      wmma_fp16_kernel_32_8_16<<<grid, WARP_SIZE>>>(dev_a, dev_b, m, n, dev_c,
                                                    k);
      break;
  }
  for (int i = 0; i < M; ++i) {
    CUDA_CHECK(cudaMemcpy(c + i * N, dev_c + i * n, N * sizeof(float),
                          cudaMemcpyDeviceToHost));
  }
  CUDA_CHECK(cudaFree(dev_a));
  CUDA_CHECK(cudaFree(dev_b));
  CUDA_CHECK(cudaFree(dev_c));
  return 0;
}
