#include "utils.h"
#include "cuda.h"
#include <mma.h>
#include "wmma/wmma_function.h"
using namespace nvcuda;
#define STRH(x) #x
#define STR(x) STRH(x)
__device__ void print_CUDA_ARCH_() {
  const static char *p = STR(__CUDA_ARCH__);
  printf("__CUDA_ARCH__:%s \n", p);
}

__global__ void print_CUDA_ARCH() { print_CUDA_ARCH_(); }
void test_row_major(TensorKenerlType t = TensorKenerl_16_16_16) {
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
  wmma_fp16(h_a, h_b, M, N, h_c, K, false, t);

  // Print the result
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      printf("%-2d ", static_cast<int>(h_c[i * N + j]));
    }
    printf("\n");
  }
}
void test_col_major(TensorKenerlType t = TensorKenerl_16_16_16) {
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

  // Print the result
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      printf("%-2d ", static_cast<int>(h_c[i * N + j]));
    }
    printf("\n");
  }
}
int main() {
  test_row_major(TensorKenerl_8_32_16);
  // printf("\n");
  // printf("-------------------------\n");
  // printf("\n");
  // test_col_major();

  return 0;
}
