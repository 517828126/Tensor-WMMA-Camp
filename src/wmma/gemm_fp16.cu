#include "wmma_function.h"
using namespace nvcuda;

template <int fragment_m, int fragment_n, int fragment_k>
__global__ void wmma_fp16_kernel(const half *a, const half *b, int M, int N,
                                 float *c, int K) {
  size_t row = fragment_m * blockIdx.y;
  size_t col = fragment_n * blockIdx.x;

  if (row >= M || col >= N) {
    return;
  }

  wmma::fragment<wmma::accumulator, fragment_m, fragment_n, fragment_k, float>
      c_frag;
  wmma::fill_fragment(c_frag, 0.0f);
  size_t loop_n = (K + fragment_k - 1) / fragment_k;
#pragma unroll
  for (size_t i = 0; i < loop_n; ++i) {
    wmma::fragment<wmma::matrix_a, fragment_m, fragment_n, fragment_k, half,
                   wmma::row_major>
        a_frag;
    wmma::fragment<wmma::matrix_b, fragment_m, fragment_n, fragment_k, half,
                   wmma::col_major>
        b_frag;
    wmma::load_matrix_sync(a_frag, a + row * K + i * fragment_k, K);
    wmma::load_matrix_sync(b_frag, b + col * K + i * fragment_k, K);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }
  wmma::store_matrix_sync(c + row * N + col, c_frag, N, wmma::mem_row_major);
}

template <int fragment_m, int fragment_n, int fragment_k>
__global__ void wmma_fp16_kernel(const half *a, const half *b, int M, int N,
                                 float *c, int K, int lda, int ldb, int ldc) {
  size_t row = fragment_m * blockIdx.y;
  size_t col = fragment_n * blockIdx.x;

  if (row >= M || col >= N) {
    return;
  }

  wmma::fragment<wmma::accumulator, fragment_m, fragment_n, fragment_k, float>
      c_frag;
  wmma::fill_fragment(c_frag, 0.0f);
  size_t loop_n = (K + fragment_k - 1) / fragment_k;
#pragma unroll
  for (size_t i = 0; i < loop_n; ++i) {
    wmma::fragment<wmma::matrix_a, fragment_m, fragment_n, fragment_k, half,
                   wmma::row_major>
        a_frag;
    wmma::fragment<wmma::matrix_b, fragment_m, fragment_n, fragment_k, half,
                   wmma::col_major>
        b_frag;
    wmma::load_matrix_sync(a_frag, a + row * lda + i * fragment_k, lda);
    wmma::load_matrix_sync(b_frag, b + col * ldb + i * fragment_k, ldb);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }
  wmma::store_matrix_sync(c + row * ldc + col, c_frag, ldc,
                          wmma::mem_row_major);
}

template <int fragment_m, int fragment_n, int fragment_k>
__global__ void wmma_fp16_kernel(const half *a, const half *b, int M, int N,
                                 half *c, int K) {
  size_t row = fragment_m * blockIdx.y;
  size_t col = fragment_n * blockIdx.x;

  if (row >= M || col >= N) {
    return;
  }

  wmma::fragment<wmma::accumulator, fragment_m, fragment_n, fragment_k, half>
      c_frag;
  wmma::fill_fragment(c_frag, 0.0f);
  size_t loop_n = (K + fragment_k - 1) / fragment_k;
#pragma unroll
  for (size_t i = 0; i < loop_n; ++i) {
    wmma::fragment<wmma::matrix_a, fragment_m, fragment_n, fragment_k, half,
                   wmma::row_major>
        a_frag;
    wmma::fragment<wmma::matrix_b, fragment_m, fragment_n, fragment_k, half,
                   wmma::col_major>
        b_frag;
    wmma::load_matrix_sync(a_frag, a + row * K + i * fragment_k, K);
    wmma::load_matrix_sync(b_frag, b + col * K + i * fragment_k, K);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }
  wmma::store_matrix_sync(c + row * N + col, c_frag, N, wmma::mem_row_major);
}

template <int fragment_m, int fragment_n, int fragment_k>
__global__ void wmma_fp16_kernel(const half *a, const half *b, int M, int N,
                                 half *c, int K, int lda, int ldb, int ldc) {
  size_t row = fragment_m * blockIdx.y;
  size_t col = fragment_n * blockIdx.x;

  if (row >= M || col >= N) {
    return;
  }

  wmma::fragment<wmma::accumulator, fragment_m, fragment_n, fragment_k, half>
      c_frag;
  wmma::fill_fragment(c_frag, 0.0f);
  size_t loop_n = (K + fragment_k - 1) / fragment_k;
#pragma unroll
  for (size_t i = 0; i < loop_n; ++i) {
    wmma::fragment<wmma::matrix_a, fragment_m, fragment_n, fragment_k, half,
                   wmma::row_major>
        a_frag;
    wmma::fragment<wmma::matrix_b, fragment_m, fragment_n, fragment_k, half,
                   wmma::col_major>
        b_frag;
    wmma::load_matrix_sync(a_frag, a + row * lda + i * fragment_k, lda);
    wmma::load_matrix_sync(b_frag, b + col * ldb + i * fragment_k, ldb);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }
  wmma::store_matrix_sync(c + row * ldc + col, c_frag, ldc,
                          wmma::mem_row_major);
}

template <typename SRC_A, typename SRC_B, typename DST_C>
int wmma_fp16_imp(const SRC_A *a, const SRC_B *b, int M, int N, DST_C *c, int K,
                  bool matrix_b_is_col_major,
                  FP16TensorKenerlType kenerl_type) {
  auto p1 = std::chrono::high_resolution_clock::now();
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
  int m;
  int n;
  int k;
  half *dev_a;
  half *dev_b;
  DST_C *dev_c;
  data_preprocess(grid, a, b, M, N, K, dev_a, dev_b, dev_c, m, n, k, kenerl_m,
                  kenerl_n, kenerl_k, matrix_b_is_col_major);
  auto p2 = std::chrono::high_resolution_clock::now();
  switch (kenerl_type) {
    case FP16TensorKenerlType::TensorKenerl_16_16_16:
      wmma_kernel<16, 16, 16>
          <<<grid, WARP_SIZE>>>(dev_a, dev_b, m, n, dev_c, k);
      break;
    case FP16TensorKenerlType::TensorKenerl_8_32_16:
      wmma_kernel<8, 32, 16><<<grid, WARP_SIZE>>>(dev_a, dev_b, m, n, dev_c, k);
      break;
    case FP16TensorKenerlType::TensorKenerl_32_8_16:
      wmma_kernel<32, 8, 16><<<grid, WARP_SIZE>>>(dev_a, dev_b, m, n, dev_c, k);
      break;
  }
  auto p3 = std::chrono::high_resolution_clock::now();
  if (M == m && N == n) {
    cudaMemcpy(c, dev_c, M * N * sizeof(DST_C), cudaMemcpyDeviceToHost);
  } else {
    for (int i = 0; i < M; ++i) {
      CUDA_CHECK(cudaMemcpy(c + i * N, dev_c + i * n, N * sizeof(DST_C),
                            cudaMemcpyDeviceToHost));
    }
  }

  CUDA_CHECK(cudaFree(dev_a));
  CUDA_CHECK(cudaFree(dev_b));
  CUDA_CHECK(cudaFree(dev_c));
  auto p4 = std::chrono::high_resolution_clock::now();
  auto cost1 =
      std::chrono::duration_cast<std::chrono::microseconds>(p2 - p1).count();
  auto cost2 =
      std::chrono::duration_cast<std::chrono::microseconds>(p3 - p2).count();
  auto cost3 =
      std::chrono::duration_cast<std::chrono::microseconds>(p4 - p3).count();
  printf("data_preprocess:%llu,kernel:%llu,result cpoy and cudafree:%llu \n",
         cost1, cost2, cost3);
  return 0;
}

template <typename SRC_A, typename SRC_B, typename DST_C>
int wmma_fp16_imp(const SRC_A *a, const SRC_B *b, int M, int N, DST_C *c, int K,
                  size_t pitch_src_a, size_t pitch_src_b, size_t pitch_src_c,
                  bool matrix_b_is_col_major,
                  FP16TensorKenerlType kenerl_type) {
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
  int m;
  int n;
  int k;
  half *dev_a;
  half *dev_b;
  DST_C *dev_c;
  size_t pitch_dev_a;
  size_t pitch_dev_b;
  size_t pitch_dev_c;
  auto p1 = std::chrono::high_resolution_clock::now();
  data_preprocess(grid, a, b, M, N, K, pitch_src_a, pitch_src_b, dev_a, dev_b,
                  dev_c, m, n, k, pitch_dev_a, pitch_dev_b, pitch_dev_c,
                  kenerl_m, kenerl_n, kenerl_k, matrix_b_is_col_major);
  int lda = pitch_dev_a / sizeof(half);
  int ldb = pitch_dev_b / sizeof(half);
  int ldc = pitch_dev_c / sizeof(DST_C);
  auto p2 = std::chrono::high_resolution_clock::now();
  switch (kenerl_type) {
    case FP16TensorKenerlType::TensorKenerl_16_16_16:
      wmma_kernel<16, 16, 16>
          <<<grid, WARP_SIZE>>>(dev_a, dev_b, m, n, dev_c, k, lda, ldb, ldc);
      break;
    case FP16TensorKenerlType::TensorKenerl_8_32_16:
      wmma_kernel<8, 32, 16>
          <<<grid, WARP_SIZE>>>(dev_a, dev_b, m, n, dev_c, k, lda, ldb, ldc);
      break;
    case FP16TensorKenerlType::TensorKenerl_32_8_16:
      wmma_kernel<32, 8, 16>
          <<<grid, WARP_SIZE>>>(dev_a, dev_b, m, n, dev_c, k, lda, ldb, ldc);
      break;
  }
  auto p3 = std::chrono::high_resolution_clock::now();
  if (M == m && N == n && pitch_dev_c == N * sizeof(DST_C)) {
    cudaMemcpy(c, dev_c, M * N * sizeof(DST_C), cudaMemcpyDeviceToHost);
  } else {
    cudaMemcpy2D(c, pitch_src_c, dev_c, pitch_dev_c, N * sizeof(DST_C), M,
                 cudaMemcpyDeviceToHost);
  }

  CUDA_CHECK(cudaFree(dev_a));
  CUDA_CHECK(cudaFree(dev_b));
  CUDA_CHECK(cudaFree(dev_c));
  auto p4 = std::chrono::high_resolution_clock::now();
  auto cost1 =
      std::chrono::duration_cast<std::chrono::microseconds>(p2 - p1).count();
  auto cost2 =
      std::chrono::duration_cast<std::chrono::microseconds>(p3 - p2).count();
  auto cost3 =
      std::chrono::duration_cast<std::chrono::microseconds>(p4 - p3).count();
  printf("data_preprocess:%llu,kernel:%llu,result cpoy and cudafree:%llu \n",
         cost1, cost2, cost3);
  return 0;
}

int wmma_fp16(const float *a, const float *b, int M, int N, float *c, int K,
              bool matrix_b_is_col_major, FP16TensorKenerlType kenerl_type) {
  return wmma_fp16_imp(a, b, M, N, c, K, matrix_b_is_col_major, kenerl_type);
}

int wmma_fp16(const half *a, const half *b, int M, int N, float *c, int K,
              bool matrix_b_is_col_major, FP16TensorKenerlType kenerl_type) {
  return wmma_fp16_imp(a, b, M, N, c, K, matrix_b_is_col_major, kenerl_type);
}

int wmma_fp16(const half *a, const half *b, int M, int N, half *c, int K,
              bool matrix_b_is_col_major, FP16TensorKenerlType kenerl_type) {
  return wmma_fp16_imp(a, b, M, N, c, K, matrix_b_is_col_major, kenerl_type);
}

int wmma_fp16(const float *a, const float *b, int M, int N, float *c, int K,
              size_t pitch_src_a, size_t pitch_src_b, size_t pitch_src_c,
              bool matrix_b_is_col_major, FP16TensorKenerlType kenerl_type) {
  return wmma_fp16_imp(a, b, M, N, c, K, pitch_src_a, pitch_src_b, pitch_src_c,
                       matrix_b_is_col_major, kenerl_type);
}
int wmma_fp16(const half *a, const half *b, int M, int N, float *c, int K,
              size_t pitch_src_a, size_t pitch_src_b, size_t pitch_src_c,
              bool matrix_b_is_col_major, FP16TensorKenerlType kenerl_type) {
  return wmma_fp16_imp(a, b, M, N, c, K, pitch_src_a, pitch_src_b, pitch_src_c,
                       matrix_b_is_col_major, kenerl_type);
}
int wmma_fp16(const half *a, const half *b, int M, int N, half *c, int K,
              size_t pitch_src_a, size_t pitch_src_b, size_t pitch_src_c,
              bool matrix_b_is_col_major, FP16TensorKenerlType kenerl_type) {
  return wmma_fp16_imp(a, b, M, N, c, K, pitch_src_a, pitch_src_b, pitch_src_c,
                       matrix_b_is_col_major, kenerl_type);
}

template <int fragment_m, int fragment_n, int fragment_k>
__global__ void wmma_kernel_T_T(const half *a, const half *b, int M, int N,
                                half *c, int K, int lda, int ldb, int ldc) {
  size_t row = fragment_m * blockIdx.y;
  size_t col = fragment_n * blockIdx.x;

  if (row >= M || col >= N) {
    return;
  }

  wmma::fragment<wmma::accumulator, fragment_m, fragment_n, fragment_k, half>
      c_frag;
  wmma::fill_fragment(c_frag, 0);
  size_t loop_n = (K + fragment_k - 1) / fragment_k;
#pragma unroll
  for (size_t i = 0; i < loop_n; ++i) {
    wmma::fragment<wmma::matrix_a, fragment_m, fragment_n, fragment_k, half,
                   wmma::row_major>
        a_frag;
    wmma::fragment<wmma::matrix_b, fragment_m, fragment_n, fragment_k, half,
                   wmma::row_major>
        b_frag;
    wmma::load_matrix_sync(a_frag, a + row * lda + i * fragment_k, lda);
    wmma::load_matrix_sync(b_frag, b + i * fragment_k * ldb + col, ldb);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }
  wmma::store_matrix_sync(c + row * ldc + col, c_frag, ldc,
                          wmma::mem_row_major);
}

int wmma_fp16_row(const half *a, const half *b, int M, int N, half *c, int K,
                  size_t pitch_src_a, size_t pitch_src_b, size_t pitch_src_c,
                  FP16TensorKenerlType kenerl_type) {
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
  half *dev_a;
  half *dev_b;
  half *dev_c;
  size_t pitch_dev_a;
  size_t pitch_dev_b;
  size_t pitch_dev_c;
  int align_m;
  int align_n;
  int align_k;

  auto p1 = std::chrono::high_resolution_clock::now();
  data_preprocess(grid, a, b, M, N, K, pitch_src_a, pitch_src_b, dev_a, dev_b,
                  dev_c, align_m, align_n, align_k, pitch_dev_a, pitch_dev_b,
                  pitch_dev_c, kenerl_m, kenerl_n, kenerl_k, true);  
  int lda = pitch_dev_a / sizeof(half);
  int ldb = pitch_dev_b / sizeof(half);
  int ldc = pitch_dev_c / sizeof(half);
  auto p2 = std::chrono::high_resolution_clock::now();
  switch (kenerl_type) {
    case FP16TensorKenerlType::TensorKenerl_16_16_16:
      wmma_kernel_T_T<16, 16, 16><<<grid, WARP_SIZE>>>(
          dev_a, dev_b, align_m, align_n, dev_c, align_k, lda, ldb, ldc);
      break;
    case FP16TensorKenerlType::TensorKenerl_8_32_16:
      wmma_kernel_T_T<8, 32, 16><<<grid, WARP_SIZE>>>(
          dev_a, dev_b, align_m, align_n, dev_c, align_k, lda, ldb, ldc);
      break;
    case FP16TensorKenerlType::TensorKenerl_32_8_16:
      wmma_kernel_T_T<32, 8, 16><<<grid, WARP_SIZE>>>(
          dev_a, dev_b, align_m, align_n, dev_c, align_k, lda, ldb, ldc);
      break;
  }
  auto p3 = std::chrono::high_resolution_clock::now();
  void *host_c = (void *)const_cast<half *>(c);
  cudaHostRegister(host_c, M * pitch_src_c, cudaHostRegisterDefault);
  if (M == align_m && N == align_n && pitch_dev_c == N * sizeof(half)) {
    cudaMemcpy(c, dev_c, M * N * sizeof(half), cudaMemcpyDeviceToHost);
  } else {
    cudaMemcpy2D(c, pitch_src_c, dev_c, pitch_dev_c, N * sizeof(half), M,
                 cudaMemcpyDeviceToHost);
  }
  cudaHostUnregister(host_c);

  CUDA_CHECK(cudaFree(dev_a));
  CUDA_CHECK(cudaFree(dev_b));
  CUDA_CHECK(cudaFree(dev_c));
  auto p4 = std::chrono::high_resolution_clock::now();
  auto cost1 =
      std::chrono::duration_cast<std::chrono::microseconds>(p2 - p1).count();
  auto cost2 =
      std::chrono::duration_cast<std::chrono::microseconds>(p3 - p2).count();
  auto cost3 =
      std::chrono::duration_cast<std::chrono::microseconds>(p4 - p3).count();
  printf("data_preprocess:%llu,kernel:%llu,result cpoy and cudafree:%llu \n",
         cost1, cost2, cost3);
  return 0;
}
