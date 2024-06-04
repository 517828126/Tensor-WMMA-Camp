#include "common.h"
#include <mma.h>
/*
    a矩阵大小为M*K，b矩阵大小为K*N，矩阵相乘后结果c矩阵大小则为M*N
    matrix_b_is_col_major:代表b矩阵是按行主序存储还是列主序存储
    kenerl_type:不同精度提供的可选片段大小不同，按矩阵长宽比选择适合的kenerl_type
*/



int DLL_PUBLIC wmma_fp16(const float *a, const float *b, int M, int N, float *c,
                         int K, bool matrix_b_is_col_major = false,
                         FP16TensorKenerlType kenerl_type =
                             FP16TensorKenerlType::TensorKenerl_16_16_16);

int DLL_PUBLIC wmma_fp16(const half *a, const half *b, int M, int N, float *c,
                         int K, bool matrix_b_is_col_major = false,
                         FP16TensorKenerlType kenerl_type =
                             FP16TensorKenerlType::TensorKenerl_16_16_16);

int DLL_PUBLIC wmma_fp16(const half *a, const half *b, int M, int N, half *c,
                         int K, bool matrix_b_is_col_major = false,
                         FP16TensorKenerlType kenerl_type =
                             FP16TensorKenerlType::TensorKenerl_16_16_16);

int DLL_PUBLIC wmma_bf16(const float *a, const float *b, int M, int N, float *c,
                         int K, bool matrix_b_is_col_major = false,
                         BF16TensorKenerlType kenerl_type =
                             BF16TensorKenerlType::TensorKenerl_16_16_16);

int DLL_PUBLIC wmma_bf16(const __nv_bfloat16 *a, const __nv_bfloat16 *b, int M,
                         int N, float *c, int K,
                         bool matrix_b_is_col_major = false,
                         BF16TensorKenerlType kenerl_type =
                             BF16TensorKenerlType::TensorKenerl_16_16_16);