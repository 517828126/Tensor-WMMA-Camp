#include "utils.h"
#include "cuda.h"
#include <mma.h>
#include <cublas_v2.h>
#include "wmma/wmma_function.h"
#include <chrono>
#include "stdio.h"
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

void test_cublas(half* a, half* b, half* c, int M, int N, int K) {
  cudaStream_t st1, st2, st3;
  cudaStreamCreate(&st1);
  cudaStreamCreate(&st2);
  cudaStreamCreate(&st3);
  half* dev_a;
  half* dev_b;
  half* dev_c;
  cudaMalloc((void**)(&dev_a), M * K * sizeof(half));
  cudaMalloc((void**)(&dev_b), K * N * sizeof(half));
  cudaMalloc((void**)(&dev_c), M * N * sizeof(half));
  cudaMemcpyAsync(dev_a, a, M * K * sizeof(half), cudaMemcpyHostToDevice, st1);
  cudaMemcpyAsync(dev_b, b, K * N * sizeof(half), cudaMemcpyHostToDevice, st2);
  cudaMemsetAsync(dev_c, 0, M * N * sizeof(half), st3);

  cublasHandle_t cublasH = NULL;
  cublasCreate(&cublasH);
  const half alpha = __float2half(1.0f);
  const half beta = __float2half(0);
  cudaStreamSynchronize(st1);
  cudaStreamSynchronize(st2);
  cudaStreamSynchronize(st3);
  cublasHgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dev_b, N,
              dev_a, K, &beta, dev_c, N);
  cudaMemcpy(c, dev_c, M * N * sizeof(half), cudaMemcpyDeviceToHost);
}
__global__ void test_complier(float* src, half* dst) { data_cast(*src, *dst); }
int main() {
  const int M = 4096;
  const int N = 4096;
  const int K = 4096;
  half* a;
  half* b;
  half* c;
  half* w;
  a = new half[M * K];
  b = new half[K * N];
  c = new half[M * N];
  w = new half[M * N];
  memset(a, 0, M * K * sizeof(half));
  memset(b, 0, K * N * sizeof(half));
  memset(c, 0, M * N * sizeof(half));
  memset(w, 0, M * N * sizeof(half));
  // Initialize original matrices with actual data
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < K; ++j) {
      a[i * K + j] = __float2half(1.0f);
    }
  }
  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < N; ++j) {
      if (i == 0) {
        b[i * N + j] = __float2half(1.0f);
      } else {
        b[i * N + j] = 0;
      }
    }
  }
  for (int i = 0; i < 5; i++) {
    size_t cost1, cost2, cost3;
    {
      auto p1 = std::chrono::high_resolution_clock::now();
      // wmma_fp16(a, b, M, N, w, K, false,
      //           FP16TensorKenerlType::TensorKenerl_16_16_16);
      wmma_fp16_row(a, b, M, N, w, K, K * sizeof(half), N * sizeof(half),
                    N * sizeof(half));
      auto p2 = std::chrono::high_resolution_clock::now();
      // wmma_fp16(a, b, M, N, w, K, K * sizeof(half), N * sizeof(half),
      //           N * sizeof(half), false,
      //           FP16TensorKenerlType::TensorKenerl_16_16_16);
      auto p3 = std::chrono::high_resolution_clock::now();
      cost1 = std::chrono::duration_cast<std::chrono::microseconds>(p2 - p1)
                  .count();
      cost2 = std::chrono::duration_cast<std::chrono::microseconds>(p3 - p2)
                  .count();
    }

    {
      auto p1 = std::chrono::high_resolution_clock::now();
      test_cublas(a, b, c, M, N, K);
      auto p2 = std::chrono::high_resolution_clock::now();
      cost3 = std::chrono::duration_cast<std::chrono::microseconds>(p2 - p1)
                  .count();
    }
    std::string cmp_ret =
        memcmp(w, c, M * N * sizeof(half)) == 0 ? "true" : "fasle";

    printf("wmma:%llu,2D wmma:%llu,cublas:%llu ,memcmp:%s\n", cost1, cost2,
           cost3, cmp_ret.c_str());
  }
  return 0;
}
