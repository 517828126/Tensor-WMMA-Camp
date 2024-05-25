#include "common.h"
#include <mma.h>
#include "wmma_function.h"
using namespace nvcuda;
int wmma_fp16_kenerl_16_16_16(const float *a, const float *b, int M, int N,
                              float *c, int K, bool matrix_b_is_col_major);
int wmma_fp16_kenerl_8_32_16(const float *a, const float *b, int M, int N,
                             float *c, int K, bool matrix_b_is_col_major);
int wmma_fp16_kenerl_32_8_16(const float *a, const float *b, int M, int N,
                             float *c, int K, bool matrix_b_is_col_major);
int wmma_fp16_kenerl_16_16_16(const half *a, const half *b, int M, int N,
                              float *c, int K, bool matrix_b_is_col_major);
int wmma_fp16_kenerl_8_32_16(const half *a, const half *b, int M, int N,
                             float *c, int K, bool matrix_b_is_col_major);
int wmma_fp16_kenerl_32_8_16(const half *a, const half *b, int M, int N,
                             float *c, int K, bool matrix_b_is_col_major);
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
              bool matrix_b_is_col_major, TensorKenerlType kenerl_type) {
  if (M < 0 || N < 0 || K < 0) {
    return -1;
  }
  switch (kenerl_type) {
    case TensorKenerl_16_16_16:
      return wmma_fp16_kenerl_16_16_16(a, b, M, N, c, K, matrix_b_is_col_major);
      break;
    case TensorKenerl_8_32_16:
      return wmma_fp16_kenerl_8_32_16(a, b, M, N, c, K, matrix_b_is_col_major);
      break;
    case TensorKenerl_32_8_16:
      return wmma_fp16_kenerl_32_8_16(a, b, M, N, c, K, matrix_b_is_col_major);
      break;
  }
  return -1;
}

int wmma_fp16_kenerl_16_16_16(const float *a, const float *b, int M, int N,
                              float *c, int K, bool matrix_b_is_col_major) {
  dim3 grid;
  half *dev_a;
  half *dev_b;
  float *dev_c;
  int m = (M + 15) / 16;
  int n = (N + 15) / 16;
  int k = (K + 15) / 16;
  m *= 16;
  n *= 16;
  k *= 16;
  grid.y = (m + 15) / 16;
  grid.x = (n + 15) / 16;
  CUDA_CHECK(cudaMalloc((void **)(&dev_a), m * k * sizeof(half)));
  CUDA_CHECK(cudaMalloc((void **)(&dev_b), k * n * sizeof(half)));
  CUDA_CHECK(cudaMalloc((void **)(&dev_c), m * n * sizeof(float)));
  cudaMemset(dev_a, 0, m * k * sizeof(half));
  cudaMemset(dev_b, 0, k * n * sizeof(half));
  cudaMemset(dev_c, 0, m * n * sizeof(float));
  int block_a = (m * k + 31) / 32;
  int block_b = (k * n + 31) / 32;
  float *dev_fa;
  float *dev_fb;
  CUDA_CHECK(cudaMalloc((void **)(&dev_fa), m * k * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **)(&dev_fb), k * n * sizeof(float)));
  cudaMemset(dev_fa, 0, m * k * sizeof(half));
  cudaMemset(dev_fb, 0, k * n * sizeof(half));
  for (int i = 0; i < M; ++i) {
    CUDA_CHECK(cudaMemcpy(dev_fa + i * k, a + i * K, K * sizeof(float),
                          cudaMemcpyHostToDevice));
  }

  matrix_cast_kernel_float2half<<<block_a, WARP_SIZE>>>(dev_fa, dev_a, m, n,
                                                        false);
  if (!matrix_b_is_col_major) {
    for (int i = 0; i < K; ++i) {
      CUDA_CHECK(cudaMemcpy(dev_fb + i * n, b + i * N, N * sizeof(float),
                            cudaMemcpyHostToDevice));
    }
    matrix_cast_kernel_float2half<<<block_b, WARP_SIZE>>>(dev_fb, dev_b, k, n,
                                                          true);
  } else {
    for (int i = 0; i < N; ++i) {
      CUDA_CHECK(cudaMemcpy(dev_fb + i * k, b + i * K, K * sizeof(float),
                            cudaMemcpyHostToDevice));
    }
    matrix_cast_kernel_float2half<<<block_b, WARP_SIZE>>>(dev_fb, dev_b, k, n,
                                                          false);
  }
  wmma_fp16_kernel_16_16_16<<<grid, WARP_SIZE>>>(dev_a, dev_b, m, n, dev_c, k);
  float *debug = (float *)malloc(m * n * sizeof(float));
  for (int i = 0; i < M; ++i) {
    CUDA_CHECK(cudaMemcpy(c + i * N, dev_c + i * n, N * sizeof(float),
                          cudaMemcpyDeviceToHost));
  }
  CUDA_CHECK(cudaFree(dev_a));
  CUDA_CHECK(cudaFree(dev_b));
  CUDA_CHECK(cudaFree(dev_c));
  return 0;
}

int wmma_fp16_kenerl_8_32_16(const float *a, const float *b, int M, int N,
                             float *c, int K, bool matrix_b_is_col_major) {
  dim3 grid;
  half *dev_a;
  half *dev_b;
  float *dev_c;
  int m = (M + 7) / 8;
  int n = (N + 31) / 32;
  int k = (K + 16) / 16;
  m *= 8;
  n *= 32;
  k *= 16;
  grid.y = (m + 7) / 8;
  grid.x = (n + 31) / 32;
  CUDA_CHECK(cudaMalloc((void **)(&dev_a), m * k * sizeof(half)));
  CUDA_CHECK(cudaMalloc((void **)(&dev_b), k * n * sizeof(half)));
  CUDA_CHECK(cudaMalloc((void **)(&dev_c), m * n * sizeof(float)));
  cudaMemset(dev_a, 0, m * k * sizeof(half));
  cudaMemset(dev_b, 0, k * n * sizeof(half));
  cudaMemset(dev_c, 0, m * n * sizeof(float));
  int block_a = (m * k + WARP_SIZE - 1) / WARP_SIZE;
  int block_b = (k * n + WARP_SIZE - 1) / WARP_SIZE;
  float *dev_fa;
  float *dev_fb;
  CUDA_CHECK(cudaMalloc((void **)(&dev_fa), m * k * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **)(&dev_fb), k * n * sizeof(float)));
  cudaMemset(dev_fa, 0, m * k * sizeof(half));
  cudaMemset(dev_fb, 0, k * n * sizeof(half));
  for (int i = 0; i < M; ++i) {
    CUDA_CHECK(cudaMemcpy(dev_fa + i * k, a + i * K, K * sizeof(float),
                          cudaMemcpyHostToDevice));
  }

  matrix_cast_kernel_float2half<<<block_a, WARP_SIZE>>>(dev_fa, dev_a, m, n,
                                                        false);
  if (!matrix_b_is_col_major) {
    for (int i = 0; i < K; ++i) {
      CUDA_CHECK(cudaMemcpy(dev_fb + i * n, b + i * N, N * sizeof(float),
                            cudaMemcpyHostToDevice));
    }
    matrix_cast_kernel_float2half<<<block_b, WARP_SIZE>>>(dev_fb, dev_b, k, n,
                                                          true);
  } else {
    for (int i = 0; i < N; ++i) {
      CUDA_CHECK(cudaMemcpy(dev_fb + i * k, b + i * K, K * sizeof(float),
                            cudaMemcpyHostToDevice));
    }
    matrix_cast_kernel_float2half<<<block_b, WARP_SIZE>>>(dev_fb, dev_b, k, n,
                                                          false);
  }
  wmma_fp16_kernel_8_32_16<<<grid, WARP_SIZE>>>(dev_a, dev_b, m, n, dev_c, k);
  float *debug = (float *)malloc(m * n * sizeof(float));
  for (int i = 0; i < M; ++i) {
    CUDA_CHECK(cudaMemcpy(c + i * N, dev_c + i * n, N * sizeof(float),
                          cudaMemcpyDeviceToHost));
  }
  CUDA_CHECK(cudaFree(dev_a));
  CUDA_CHECK(cudaFree(dev_b));
  CUDA_CHECK(cudaFree(dev_c));
  return 0;
}
int wmma_fp16_kenerl_32_8_16(const float *a, const float *b, int M, int N,
                             float *c, int K, bool matrix_b_is_col_major) {
  dim3 grid;
  half *dev_a;
  half *dev_b;
  float *dev_c;
  int m = (M + 31) / 32;
  int n = (N + 7) / 8;
  int k = (K + 15) / 16;
  m *= 32;
  n *= 8;
  k *= 16;
  grid.y = (m + 31) / 32;
  grid.x = (n + 7) / 8;
  CUDA_CHECK(cudaMalloc((void **)(&dev_a), m * k * sizeof(half)));
  CUDA_CHECK(cudaMalloc((void **)(&dev_b), k * n * sizeof(half)));
  CUDA_CHECK(cudaMalloc((void **)(&dev_c), m * n * sizeof(float)));
  cudaMemset(dev_a, 0, m * k * sizeof(half));
  cudaMemset(dev_b, 0, k * n * sizeof(half));
  cudaMemset(dev_c, 0, m * n * sizeof(float));
  int block_a = (m * k + WARP_SIZE - 1) / WARP_SIZE;
  int block_b = (k * n + WARP_SIZE - 1) / WARP_SIZE;
  float *dev_fa;
  float *dev_fb;
  CUDA_CHECK(cudaMalloc((void **)(&dev_fa), m * k * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **)(&dev_fb), k * n * sizeof(float)));
  cudaMemset(dev_fa, 0, m * k * sizeof(half));
  cudaMemset(dev_fb, 0, k * n * sizeof(half));
  for (int i = 0; i < M; ++i) {
    CUDA_CHECK(cudaMemcpy(dev_fa + i * k, a + i * K, K * sizeof(float),
                          cudaMemcpyHostToDevice));
  }
  matrix_cast_kernel_float2half<<<block_a, WARP_SIZE>>>(dev_fa, dev_a, m, n,
                                                        false);
  if (!matrix_b_is_col_major) {
    for (int i = 0; i < K; ++i) {
      CUDA_CHECK(cudaMemcpy(dev_fb + i * n, b + i * N, N * sizeof(float),
                            cudaMemcpyHostToDevice));
    }
    matrix_cast_kernel_float2half<<<block_b, WARP_SIZE>>>(dev_fb, dev_b, k, n,
                                                          true);
  } else {
    for (int i = 0; i < N; ++i) {
      CUDA_CHECK(cudaMemcpy(dev_fb + i * k, b + i * K, K * sizeof(float),
                            cudaMemcpyHostToDevice));
    }
    matrix_cast_kernel_float2half<<<block_b, WARP_SIZE>>>(dev_fb, dev_b, k, n,
                                                          false);
  }
  wmma_fp16_kernel_32_8_16<<<grid, WARP_SIZE>>>(dev_a, dev_b, m, n, dev_c, k);
  float *debug = (float *)malloc(m * n * sizeof(float));
  for (int i = 0; i < M; ++i) {
    CUDA_CHECK(cudaMemcpy(c + i * N, dev_c + i * n, N * sizeof(float),
                          cudaMemcpyDeviceToHost));
  }
  CUDA_CHECK(cudaFree(dev_a));
  CUDA_CHECK(cudaFree(dev_b));
  CUDA_CHECK(cudaFree(dev_c));
  return 0;
}

int wmma_fp16(const half *a, const half *b, int M, int N, float *c, int K,
              bool matrix_b_is_col_major, TensorKenerlType kenerl_type) {
  if (M < 0 || N < 0 || K < 0) {
    return -1;
  }
  switch (kenerl_type) {
    case TensorKenerl_16_16_16:
      return wmma_fp16_kenerl_16_16_16(a, b, M, N, c, K, matrix_b_is_col_major);
      break;
    case TensorKenerl_8_32_16:
      return wmma_fp16_kenerl_8_32_16(a, b, M, N, c, K, matrix_b_is_col_major);
      break;
    case TensorKenerl_32_8_16:
      return wmma_fp16_kenerl_32_8_16(a, b, M, N, c, K, matrix_b_is_col_major);
      break;
  }
  return -1;
}

int wmma_fp16_kenerl_16_16_16(const half *a, const half *b, int M, int N,
                              float *c, int K, bool matrix_b_is_col_major) {
  dim3 grid;
  half *dev_a;
  half *dev_b;
  float *dev_c;
  int m = (M + 15) / 16;
  int n = (N + 15) / 16;
  int k = (K + 15) / 16;
  m *= 16;
  n *= 16;
  k *= 16;
  grid.y = (m + 15) / 16;
  grid.x = (n + 15) / 16;
  CUDA_CHECK(cudaMalloc((void **)(&dev_a), m * k * sizeof(half)));
  CUDA_CHECK(cudaMalloc((void **)(&dev_b), k * n * sizeof(half)));
  CUDA_CHECK(cudaMalloc((void **)(&dev_c), m * n * sizeof(float)));
  cudaMemset(dev_a, 0, m * k * sizeof(half));
  cudaMemset(dev_b, 0, k * n * sizeof(half));
  cudaMemset(dev_c, 0, m * n * sizeof(float));
  for (int i = 0; i < M; ++i) {
    CUDA_CHECK(cudaMemcpy(dev_a + i * k, a + i * K, K * sizeof(half),
                          cudaMemcpyHostToDevice));
  }
  if (!matrix_b_is_col_major) {
    half *temp;
    CUDA_CHECK(cudaMalloc((void **)(&temp), k * n * sizeof(half)));
    cudaMemset(temp, 0, k * n * sizeof(half));
    for (int i = 0; i < K; ++i) {
      CUDA_CHECK(cudaMemcpy(temp + i * n, b + i * N, N * sizeof(float),
                            cudaMemcpyHostToDevice));
    }
    int block_b = (k * n + 31) / 32;
    matrix_transpose<<<block_b, WARP_SIZE>>>(temp, dev_b, k, n);
  } else {
    for (int i = 0; i < N; ++i) {
      CUDA_CHECK(cudaMemcpy(dev_b + i * k, b + i * K, K * sizeof(float),
                            cudaMemcpyHostToDevice));
    }
  }

  wmma_fp16_kernel_16_16_16<<<grid, 32>>>(dev_a, dev_b, m, n, dev_c, k);
  float *debug = (float *)malloc(m * n * sizeof(float));
  for (int i = 0; i < M; ++i) {
    CUDA_CHECK(cudaMemcpy(c + i * N, dev_c + i * n, N * sizeof(float),
                          cudaMemcpyDeviceToHost));
  }
  CUDA_CHECK(cudaFree(dev_a));
  CUDA_CHECK(cudaFree(dev_b));
  CUDA_CHECK(cudaFree(dev_c));
  return 0;
}
int wmma_fp16_kenerl_8_32_16(const half *a, const half *b, int M, int N,
                             float *c, int K, bool matrix_b_is_col_major) {
  dim3 grid;
  half *dev_a;
  half *dev_b;
  float *dev_c;
  int m = (M + 7) / 8;
  int n = (N + 31) / 32;
  int k = (K + 16) / 16;
  m *= 8;
  n *= 32;
  k *= 16;
  grid.y = (m + 7) / 8;
  grid.x = (n + 31) / 32;
  CUDA_CHECK(cudaMalloc((void **)(&dev_a), m * k * sizeof(half)));
  CUDA_CHECK(cudaMalloc((void **)(&dev_b), k * n * sizeof(half)));
  CUDA_CHECK(cudaMalloc((void **)(&dev_c), m * n * sizeof(float)));
  cudaMemset(dev_a, 0, m * k * sizeof(half));
  cudaMemset(dev_b, 0, k * n * sizeof(half));
  cudaMemset(dev_c, 0, m * n * sizeof(float));

  for (int i = 0; i < M; ++i) {
    CUDA_CHECK(cudaMemcpy(dev_a + i * k, a + i * K, K * sizeof(float),
                          cudaMemcpyHostToDevice));
  }

  if (!matrix_b_is_col_major) {
    int block_b = (k * n + WARP_SIZE - 1) / WARP_SIZE;
    half *temp;
    CUDA_CHECK(cudaMalloc((void **)(&temp), k * n * sizeof(float)));
    cudaMemset(temp, 0, k * n * sizeof(half));    
    for (int i = 0; i < K; ++i) {
      CUDA_CHECK(cudaMemcpy(temp + i * n, b + i * N, N * sizeof(float),
                            cudaMemcpyHostToDevice));
    }
    matrix_transpose<<<block_b, WARP_SIZE>>>(temp, dev_b, k, n);
  } else {
    for (int i = 0; i < N; ++i) {
      CUDA_CHECK(cudaMemcpy(dev_b + i * k, b + i * K, K * sizeof(float),
                            cudaMemcpyHostToDevice));
    }
  }
  wmma_fp16_kernel_8_32_16<<<grid, WARP_SIZE>>>(dev_a, dev_b, m, n, dev_c, k);
  float *debug = (float *)malloc(m * n * sizeof(float));
  for (int i = 0; i < M; ++i) {
    CUDA_CHECK(cudaMemcpy(c + i * N, dev_c + i * n, N * sizeof(float),
                          cudaMemcpyDeviceToHost));
  }
  CUDA_CHECK(cudaFree(dev_a));
  CUDA_CHECK(cudaFree(dev_b));
  CUDA_CHECK(cudaFree(dev_c));
  return 0;                              
  return 0;
}
int wmma_fp16_kenerl_32_8_16(const half *a, const half *b, int M, int N,
                             float *c, int K, bool matrix_b_is_col_major) {
  return 0;
}