#include "common.h"
#include <mma.h>

int DLL_PUBLIC wmma_fp16(const float *a, const float *b, int M, int N, float *c,
                         int K, bool matrix_b_is_col_major = false,
                         FP16TensorKenerlType kenerl_type = TensorKenerl_16_16_16);

int DLL_PUBLIC wmma_fp16(const half *a, const half *b, int M, int N, float *c,
                         int K, bool matrix_b_is_col_major = false,
                         FP16TensorKenerlType kenerl_type = TensorKenerl_16_16_16);