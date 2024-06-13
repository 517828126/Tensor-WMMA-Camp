#include "utils.h"
#include "cuda.h"
#include <mma.h>
#include <cublas_v2.h>
#include "wmma/wmma_function.h"
#include <chrono>
using namespace nvcuda;
#define STRH(x) #x
#define STR(x) STRH(x)
void printMat(float* src, int M, int N) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      printf("%-2d ", static_cast<int>(src[i * N + j]));
    }
    printf("\n");
  }
}

void printMat(half* src, int M, int N) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      printf("%-2d ", static_cast<int>(__half2float(src[i * N + j])));
    }
    printf("\n");
  }
}

__device__ void print_CUDA_ARCH_() {
  const static char* p = STR(__CUDA_ARCH__);
  printf("__CUDA_ARCH__:%s \n", p);
}

__global__ void print_CUDA_ARCH() { print_CUDA_ARCH_(); }
void test_row_major(
    FP16TensorKenerlType t = FP16TensorKenerlType::TensorKenerl_16_16_16) {
  const int M = 15;
  const int N = 17;
  const int K = 17;
  float h_a[M * K] = {0};
  float h_b[K * N] = {0};
  float h_c[M * N] = {0};

  // Initialize original matrices with actual data
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < K; ++j) {
      h_a[i * K + j] = 1;
    }
  }
  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < N; ++j) {
      if (i == 0) {
        h_b[i * N + j] = j;
      } else {
        h_b[i * N + j] = 0;
      }
    }
  }

  /*
   *   B = | 0.0 | 1.0 | 2.0 | 3.0 | 4.0 | ...
   *       | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | ...
   *       | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | ...
   *       | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | ...
   *       | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | ...
   *       | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | ...
   *       |  .  |  .  |  .  |  .  |  .  | ...
   *       |  .  |  .  |  .  |  .  |  .  | ...
   *       |  .  |  .  |  .  |  .  |  .  | ...
   */
  wmma_fp16(h_a, h_b, M, N, h_c, K, false, t);

  // Print the result
  printMat(h_c, M, N);
}
void test_col_major(
    FP16TensorKenerlType t = FP16TensorKenerlType::TensorKenerl_16_16_16) {
  const int M = 15;
  const int N = 17;
  const int K = 17;
  float h_a[M * K] = {0};
  float h_b[K * N] = {0};
  float h_c[M * N] = {0};

  // Initialize original matrices with actual data
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < K; ++j) {
      h_a[i * K + j] = 1;
    }
  }

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < K; ++j) {
      if (i == 0) {
        h_b[i * K + j] = j;
      } else {
        h_b[i * K + j] = 0;
      }
    }
  }
  /*
   *   B = | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | ...
   *       | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | ...
   *       | 2.0 | 0.0 | 0.0 | 0.0 | 0.0 | ...
   *       | 3.0 | 0.0 | 0.0 | 0.0 | 0.0 | ...
   *       | 4.0 | 0.0 | 0.0 | 0.0 | 0.0 | ...
   *       | 5.0 | 0.0 | 0.0 | 0.0 | 0.0 | ...
   *       |  .  |  .  |  .  |  .  |  .  | ...
   *       |  .  |  .  |  .  |  .  |  .  | ...
   *       |  .  |  .  |  .  |  .  |  .  | ...
   */

  wmma_fp16(h_a, h_b, M, N, h_c, K, true, t);
  printMat(h_c, M, N);
}
void test_row_major_bf16(
    BF16TensorKenerlType t = BF16TensorKenerlType::TensorKenerl_16_16_16) {
  const int M = 16;
  const int N = 16;
  const int K = 16;
  float h_a[M * K] = {0};
  float h_b[K * N] = {0};
  float h_c[M * N] = {0};

  // Initialize original matrices with actual data
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < K; ++j) {
      h_a[i * K + j] = 1;
    }
  }
  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < N; ++j) {
      h_b[i * N + j] = 1;
    }
  }
  /*
   *   B = | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | ...
   *       | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | ...
   *       | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | ...
   *       | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | ...
   *       | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | ...
   *       | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | ...
   *       |  .  |  .  |  .  |  .  |  .  | ...
   *       |  .  |  .  |  .  |  .  |  .  | ...
   *       |  .  |  .  |  .  |  .  |  .  | ...
   */
  wmma_bf16(h_a, h_b, M, N, h_c, K, false, t);

  // Print the result
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      printf("%-2d ", static_cast<int>(h_c[i * N + j]));
    }
    printf("\n");
  }
}

void test_cublas(float* a, float* b, float* c, int M, int N, int K) {
  float* dev_a;
  float* dev_b;
  float* dev_c;
  cudaMalloc((void**)(&dev_a), M * K * sizeof(float));
  cudaMalloc((void**)(&dev_b), K * N * sizeof(float));
  cudaMalloc((void**)(&dev_c), M * N * sizeof(float));
  cudaMemcpy(dev_a, a, M * K * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b, K * N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(dev_c, 0, M * N * sizeof(float));

  cublasHandle_t cublasH = NULL;
  cublasCreate(&cublasH);
  const float alpha = 1.0f;
  const float beta = 0.0f;
  cublasSgemm_v2(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dev_b, N,
                 dev_a, K, &beta, dev_c, N);
  cudaMemcpy(c, dev_c, M * N * sizeof(float), cudaMemcpyDeviceToHost);
}
__global__ void test_complier(float* src, half* dst) { data_cast(*src, *dst); }
int main() {
  test_row_major();
  printf("\n");
  printf("-------------------------\n");
  printf("\n");
  test_col_major();

  return 0;
}
