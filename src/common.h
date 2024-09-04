#ifndef COMMON_H
#define COMMON_H
#include "utils.h"
#include <mma.h>
template <int fragment_m, int fragment_n, int fragment_k, typename SRC_A,
          typename SRC_B, typename DST_C>
__global__ void wmma_kernel(const SRC_A* a, const SRC_B* b, int M, int N,
                            DST_C* c, int K) {
  size_t row = fragment_m * blockIdx.y;
  size_t col = fragment_n * blockIdx.x;

  if (row >= M || col >= N) {
    return;
  }

  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, fragment_m, fragment_n,
                         fragment_k, DST_C>
      c_frag;
  nvcuda::wmma::fill_fragment(c_frag, 0.0f);
  size_t loop_n = (K + fragment_k - 1) / fragment_k;
#pragma unroll
  for (size_t i = 0; i < loop_n; ++i) {
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, fragment_m, fragment_n,
                           fragment_k, SRC_A, nvcuda::wmma::row_major>
        a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, fragment_m, fragment_n,
                           fragment_k, SRC_B, nvcuda::wmma::col_major>
        b_frag;
    nvcuda::wmma::load_matrix_sync(a_frag, a + row * K + i * fragment_k, K);
    nvcuda::wmma::load_matrix_sync(b_frag, b + col * K + i * fragment_k, K);
    nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }
  nvcuda::wmma::store_matrix_sync(c + row * N + col, c_frag, N,
                                  nvcuda::wmma::mem_row_major);
}

template <int fragment_m, int fragment_n, int fragment_k, typename SRC_A,
          typename SRC_B, typename DST_C>
__global__ void wmma_kernel(const SRC_A* a, const SRC_B* b, int M, int N,
                            DST_C* c, int K, int lda, int ldb, int ldc) {
  size_t row = fragment_m * blockIdx.y;
  size_t col = fragment_n * blockIdx.x;

  if (row >= M || col >= N) {
    return;
  }

  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, fragment_m, fragment_n,
                         fragment_k, DST_C>
      c_frag;
  nvcuda::wmma::fill_fragment(c_frag, 0.0f);
  size_t loop_n = (K + fragment_k - 1) / fragment_k;
#pragma unroll
  for (size_t i = 0; i < loop_n; ++i) {
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, fragment_m, fragment_n,
                           fragment_k, SRC_A, nvcuda::wmma::row_major>
        a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, fragment_m, fragment_n,
                           fragment_k, SRC_B, nvcuda::wmma::col_major>
        b_frag;
    nvcuda::wmma::load_matrix_sync(a_frag, a + row * lda + i * fragment_k, lda);
    nvcuda::wmma::load_matrix_sync(b_frag, b + col * ldb + i * fragment_k, ldb);
    nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }
  nvcuda::wmma::store_matrix_sync(c + row * ldc + col, c_frag, ldc,
                                  nvcuda::wmma::mem_row_major);
}

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
  // const A* src_row = (const A*)((const char*)src + i * spitch);
  // if (transpose) {
  //   B* dst_row = (B*)((char*)dst + j * dpitch);
  //   data_cast(src_row[j], dst_row[i]);
  // } else {
  //   B* dst_row = (B*)((char*)dst + i * dpitch);
  //   data_cast(src_row[j], dst_row[j]);
  // }
  if (transpose) {
    // dst[i][j] = src[j][i];
    const A* src_row = (const A*)((const char*)src + j * spitch);
    B* dst_row = (B*)((char*)dst + i * dpitch);
    data_cast(src_row[i], dst_row[j]);
  } else {
    // dst[i][j] = src[i][j];
    const A* src_row = (const A*)((const char*)src + i * spitch);
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
  // const A* src_row = (const A*)((const char*)src + i * spitch);
  // if (transpose) {
  //   A* dst_row = (A*)((char*)dst + j * dpitch);
  //   dst_row[i] = src_row[j];
  // } else {
  //   A* dst_row = (A*)((char*)dst + i * dpitch);
  //   dst_row[j] = src_row[j];
  // }

  if (transpose) {
    // dst[i][j] = src[j][i];
    const A* src_row = (const A*)((const char*)src + j * spitch);
    A* dst_row = (A*)((char*)dst + i * dpitch);
    dst_row[j] = src_row[i];
  } else {
    // dst[i][j] = src[i][j];
    const A* src_row = (const A*)((const char*)src + i * spitch);
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
  auto p1 = std::chrono::high_resolution_clock::now();
  cudaStream_t st1, st2, st3;
  cudaStreamCreate(&st1);
  cudaStreamCreate(&st2);
  cudaStreamCreate(&st3);
  void* host_a = (void*)const_cast<SRC_A*>(a);
  void* host_b = (void*)const_cast<SRC_B*>(b);
  cudaHostRegister(host_a, M * K * sizeof(SRC_A), cudaHostRegisterDefault);
  cudaHostRegister(host_b, K * N * sizeof(SRC_B), cudaHostRegisterDefault);
  CUDA_CHECK(cudaMalloc((void**)(&dev_a), m * k * sizeof(DEV_A)));
  CUDA_CHECK(cudaMalloc((void**)(&dev_b), k * n * sizeof(DEV_B)));
  CUDA_CHECK(cudaMalloc((void**)(&dev_c), m * n * sizeof(DEV_C)));
  // cudaMemset(dev_a, 0, m * k * sizeof(DEV_A));
  // cudaMemset(dev_b, 0, k * n * sizeof(DEV_B));
  // cudaMemset(dev_c, 0, m * n * sizeof(DEV_C));
  cudaMemsetAsync(dev_a, 0, m * k * sizeof(DEV_A), st1);
  cudaMemsetAsync(dev_b, 0, k * n * sizeof(DEV_B), st2);
  cudaMemsetAsync(dev_c, 0, m * n * sizeof(DEV_C), st3);
  auto p2 = std::chrono::high_resolution_clock::now();
  SRC_A* dev_src_a = nullptr;
  SRC_B* dev_src_b = nullptr;
  if (std::is_same_v<SRC_A, DEV_A> && (M == m && K == k)) {
    // CUDA_CHECK(cudaMemcpy(dev_a, a, m * k * sizeof(SRC_A),
    // cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyAsync(dev_a, a, m * k * sizeof(SRC_A),
                               cudaMemcpyHostToDevice, st1));
  } else {
    CUDA_CHECK(cudaMalloc((void**)(&dev_src_a), M * K * sizeof(SRC_A)));
    // CUDA_CHECK(cudaMemcpy(dev_src_a, a, M * K *
    // sizeof(SRC_A),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyAsync(dev_src_a, a, M * K * sizeof(SRC_A),
                               cudaMemcpyHostToDevice, st1));
    int block = ((M * K + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    data_preprocess_kernel<<<block, WARP_SIZE, 0, st1>>>(
        dev_a, k * sizeof(DEV_A), dev_src_a, K * sizeof(SRC_A), M, K, false);
  }
  auto p3 = std::chrono::high_resolution_clock::now();
  if (std::is_same_v<SRC_B, DEV_B> && (K == k && N == n) &&
      matrix_b_is_col_major) {
    // CUDA_CHECK(cudaMemcpy(dev_b, b, k * n *
    // sizeof(DEV_B),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyAsync(dev_b, b, k * n * sizeof(DEV_B),
                               cudaMemcpyHostToDevice, st2));
  } else {
    CUDA_CHECK(cudaMalloc((void**)(&dev_src_b), K * N * sizeof(SRC_B)));
    // CUDA_CHECK(cudaMemcpy(dev_src_b, b, K * N *
    // sizeof(SRC_B),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyAsync(dev_src_b, b, K * N * sizeof(SRC_B),
                               cudaMemcpyHostToDevice, st2));
    int block = ((K * N + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    if (matrix_b_is_col_major) {
      data_preprocess_kernel<<<block, WARP_SIZE, 0, st2>>>(
          dev_b, n * sizeof(DEV_B), dev_src_b, N * sizeof(SRC_B), K, N, false);
    } else {
      data_preprocess_kernel<<<block, WARP_SIZE, 0, st2>>>(
          dev_b, k * sizeof(DEV_B), dev_src_b, N * sizeof(SRC_B), K, N, true);
    }
  }
  cudaStreamSynchronize(st1);
  cudaStreamSynchronize(st2);
  cudaStreamSynchronize(st3);
  if (dev_src_a != nullptr) {
    CUDA_CHECK(cudaFree(dev_src_a));
  }
  if (dev_src_b != nullptr) {
    CUDA_CHECK(cudaFree(dev_src_b));
  }
  cudaHostUnregister(host_a);
  cudaHostUnregister(host_b);
  auto p4 = std::chrono::high_resolution_clock::now();
  auto cost1 =
      std::chrono::duration_cast<std::chrono::microseconds>(p2 - p1).count();
  auto cost2 =
      std::chrono::duration_cast<std::chrono::microseconds>(p3 - p2).count();
  auto cost3 =
      std::chrono::duration_cast<std::chrono::microseconds>(p4 - p3).count();
  printf("malloc and memset:%llu,preprocess_a:%llu,preprocess_b:%llu\n", cost1,
         cost2, cost3);
}

template <typename SRC, typename DEV>
void data_pre(const SRC* src, size_t pitch_src, size_t height_src,
              size_t width_src, DEV*& dev, size_t& pitch_dev, size_t height_dev,
              size_t width_dev, bool trans, SRC*& dev_src, cudaStream_t st) {
  if (false == trans) {
    CUDA_CHECK(cudaMallocPitch((void**)&dev, &pitch_dev,
                               width_dev * sizeof(DEV), height_dev));
    if (pitch_dev % sizeof(DEV) == 0) {
      if (std::is_same_v<SRC, DEV> &&
          (width_dev == width_src && height_dev == height_src)) {
        CUDA_CHECK(cudaMemcpy2DAsync(dev, pitch_dev, src, pitch_src,
                                     width_src * sizeof(SRC), height_src,
                                     cudaMemcpyHostToDevice, st));
      } else {
        CUDA_CHECK(cudaMemset2DAsync(dev, pitch_dev, 0, width_dev * sizeof(DEV),
                                     height_dev, st));
        size_t pitch_dev_src;
        CUDA_CHECK(cudaMallocPitch((void**)&dev_src, &pitch_dev_src,
                                   width_src * sizeof(SRC), height_src));
        CUDA_CHECK(cudaMemcpy2DAsync(dev_src, pitch_dev_src, src, pitch_src,
                                     width_src * sizeof(SRC), height_src,
                                     cudaMemcpyHostToDevice, st));
        int block =
            ((height_src * width_src + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
        data_preprocess_kernel<<<block, WARP_SIZE, 0, st>>>(
            dev, pitch_dev, dev_src, pitch_dev_src, height_src, width_src,
            false);
      }
    } else {
      CUDA_CHECK(cudaFree(dev));
      CUDA_CHECK(
          cudaMalloc((void**)&dev, height_dev * width_dev * sizeof(DEV)));

      pitch_dev = width_dev * sizeof(DEV);
      if (std::is_same_v<SRC, DEV> &&
          (height_dev == height_src && width_dev == width_src)) {
        CUDA_CHECK(cudaMemcpyAsync(dev, src,
                                   height_dev * width_dev * sizeof(DEV),
                                   cudaMemcpyHostToDevice, st));
      } else {
        CUDA_CHECK(
            cudaMemsetAsync(dev, 0, height_dev * width_dev * sizeof(DEV), st));
        size_t pitch_dev_src;
        CUDA_CHECK(cudaMallocPitch((void**)&dev_src, &pitch_dev_src,
                                   width_src * sizeof(SRC), height_src));
        CUDA_CHECK(cudaMemcpy2DAsync(dev_src, pitch_dev_src, src, pitch_src,
                                     width_src * sizeof(SRC), height_src,
                                     cudaMemcpyHostToDevice, st));
        int block =
            ((height_src * width_src + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
        data_preprocess_kernel<<<block, WARP_SIZE, 0, st>>>(
            dev, pitch_dev, dev_src, pitch_dev_src, height_src, width_src,
            false);
      }
    }
  } else {
    CUDA_CHECK(cudaMallocPitch((void**)&dev, &pitch_dev,
                               height_dev * sizeof(DEV), width_dev));
    CUDA_CHECK(cudaMemset2DAsync(dev, pitch_dev, 0, height_dev * sizeof(DEV),
                                 width_dev, st));
    if (pitch_dev % sizeof(DEV) != 0) {
      CUDA_CHECK(cudaFree(dev));
      CUDA_CHECK(
          cudaMalloc((void**)&dev, width_dev * height_dev * sizeof(DEV)));
      CUDA_CHECK(
          cudaMemsetAsync(dev, 0, width_dev * height_dev * sizeof(DEV), st));
      pitch_dev = height_dev * sizeof(DEV);
    }
    size_t pitch_dev_src;
    CUDA_CHECK(cudaMallocPitch((void**)&dev_src, &pitch_dev_src,
                               width_src * sizeof(SRC), height_src));
    CUDA_CHECK(cudaMemcpy2DAsync(dev_src, pitch_dev_src, src, pitch_src,
                                 width_src * sizeof(SRC), height_src,
                                 cudaMemcpyHostToDevice, st));
    int block =
        ((height_src * width_src + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    data_preprocess_kernel<<<block, WARP_SIZE, 0, st>>>(
        dev, pitch_dev, dev_src, pitch_dev_src, height_src, width_src, true);
  }
}

template <typename SRC, typename DEV>
void data_pre(const SRC* src, size_t pitch_src, size_t height_src,
              size_t width_src, DEV*& dev, size_t& pitch_dev, size_t height_dev,
              size_t width_dev, bool trans) {
  if (false == trans) {
    CUDA_CHECK(cudaMallocPitch((void**)&dev, &pitch_dev,
                               width_dev * sizeof(DEV), height_dev));
    if (pitch_dev % sizeof(DEV) == 0) {
      CUDA_CHECK(
          cudaMemset2D(dev, pitch_dev, 0, width_dev * sizeof(DEV), height_dev));
      if (std::is_same_v<SRC, DEV> &&
          (width_dev == width_src && height_dev == height_src)) {
        CUDA_CHECK(cudaMemcpy2D(dev, pitch_dev, src, pitch_src,
                                width_src * sizeof(SRC), height_src,
                                cudaMemcpyHostToDevice));
      } else {
        SRC* dev_src;
        size_t pitch_dev_src;
        CUDA_CHECK(cudaMallocPitch((void**)&dev_src, &pitch_dev_src,
                                   width_src * sizeof(SRC), height_src));
        CUDA_CHECK(cudaMemcpy2D(dev_src, pitch_dev_src, src, pitch_src,
                                width_src * sizeof(SRC), height_src,
                                cudaMemcpyHostToDevice));
        int block =
            ((height_src * width_src + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
        data_preprocess_kernel<<<block, WARP_SIZE>>>(dev, pitch_dev, dev_src,
                                                     pitch_dev_src, height_src,
                                                     width_src, false);
        CUDA_CHECK(cudaFree(dev_src));
      }
    } else {
      CUDA_CHECK(cudaFree(dev));
      CUDA_CHECK(
          cudaMalloc((void**)&dev, height_dev * width_dev * sizeof(DEV)));
      CUDA_CHECK(cudaMemset(dev, 0, height_dev * width_dev * sizeof(DEV)));
      pitch_dev = width_dev * sizeof(DEV);
      if (std::is_same_v<SRC, DEV> &&
          (height_dev == height_src && width_dev == width_src)) {
        CUDA_CHECK(cudaMemcpy(dev, src, height_dev * width_dev * sizeof(DEV),
                              cudaMemcpyHostToDevice));
      } else {
        SRC* dev_src;
        size_t pitch_dev_src;
        CUDA_CHECK(cudaMallocPitch((void**)&dev_src, &pitch_dev_src,
                                   width_src * sizeof(SRC), height_src));
        CUDA_CHECK(cudaMemcpy2D(dev_src, pitch_dev_src, src, pitch_src,
                                width_src * sizeof(SRC), height_src,
                                cudaMemcpyHostToDevice));
        int block =
            ((height_src * width_src + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
        data_preprocess_kernel<<<block, WARP_SIZE>>>(dev, pitch_dev, dev_src,
                                                     pitch_dev_src, height_src,
                                                     width_src, false);
        CUDA_CHECK(cudaFree(dev_src));
      }
    }
  } else {
    CUDA_CHECK(cudaMallocPitch((void**)&dev, &pitch_dev,
                               height_dev * sizeof(DEV), width_dev));
    CUDA_CHECK(
        cudaMemset2D(dev, pitch_dev, 0, height_dev * sizeof(DEV), width_dev));
    if (pitch_dev % sizeof(DEV) != 0) {
      CUDA_CHECK(cudaFree(dev));
      CUDA_CHECK(
          cudaMalloc((void**)&dev, width_dev * height_dev * sizeof(DEV)));
      CUDA_CHECK(cudaMemset(dev, 0, width_dev * height_dev * sizeof(DEV)));
      pitch_dev = height_dev * sizeof(DEV);
    }
    SRC* dev_src;
    size_t pitch_dev_src;
    CUDA_CHECK(cudaMallocPitch((void**)&dev_src, &pitch_dev_src,
                               width_src * sizeof(SRC), height_src));
    CUDA_CHECK(cudaMemcpy2D(dev_src, pitch_dev_src, src, pitch_src,
                            width_src * sizeof(SRC), height_src,
                            cudaMemcpyHostToDevice));
    int block =
        ((height_src * width_src + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    data_preprocess_kernel<<<block, WARP_SIZE>>>(
        dev, pitch_dev, dev_src, pitch_dev_src, height_src, width_src, true);
    CUDA_CHECK(cudaFree(dev_src));
  }
}

template <typename SRC_A, typename SRC_B, typename DEV_A, typename DEV_B,
          typename DEV_C>
void data_preprocess(dim3& grid, const SRC_A* a, const SRC_B* b, const int M,
                     const int N, const int K, size_t pitch_src_a,
                     size_t pitch_src_b, DEV_A*& dev_a, DEV_B*& dev_b,
                     DEV_C*& dev_c, int& align_m, int& align_n, int& align_k,
                     size_t& pitch_dev_a, size_t& pitch_dev_b,
                     size_t& pitch_dev_c, int kenerl_m, int kenerl_n,
                     int kenerl_k, bool matrix_b_is_col_major) {
  auto p1 = std::chrono::high_resolution_clock::now();
  align_m = (M + kenerl_m - 1) / kenerl_m;
  align_n = (N + kenerl_n - 1) / kenerl_n;
  align_k = (K + kenerl_k - 1) / kenerl_k;
  grid.y = align_m;
  grid.x = align_n;
  align_m *= kenerl_m;
  align_n *= kenerl_n;
  align_k *= kenerl_k;
  cudaStream_t st1, st2;
  cudaStreamCreate(&st1);
  cudaStreamCreate(&st2);
  void* host_a = (void*)const_cast<SRC_A*>(a);
  void* host_b = (void*)const_cast<SRC_B*>(b);
  cudaHostRegister(host_a, M * pitch_src_a, cudaHostRegisterDefault);
  cudaHostRegister(host_b, K * pitch_src_b, cudaHostRegisterDefault);

  SRC_A* dev_src_a = nullptr;
  data_pre(a, pitch_src_a, M, K, dev_a, pitch_dev_a, align_m, align_k, false,
           dev_src_a, st1);
  auto p2 = std::chrono::high_resolution_clock::now();
  SRC_B* dev_src_b = nullptr;
  data_pre(b, pitch_src_b, K, N, dev_b, pitch_dev_b, align_k, align_n,
           !matrix_b_is_col_major, dev_src_b, st2);
  auto p3 = std::chrono::high_resolution_clock::now();
  CUDA_CHECK(cudaMallocPitch((void**)&dev_c, &pitch_dev_c,
                             align_n * sizeof(DEV_C), align_m));
  if (pitch_dev_c % sizeof(DEV_C) != 0) {
    CUDA_CHECK(cudaFree(dev_c));
    CUDA_CHECK(cudaMalloc((void**)&dev_c, align_m * align_n * sizeof(DEV_C)));
    pitch_dev_c = align_n * sizeof(DEV_C);
  }
  auto p4 = std::chrono::high_resolution_clock::now();
  cudaStreamSynchronize(st1);
  cudaStreamSynchronize(st2);
  cudaStreamDestroy(st1);
  cudaStreamDestroy(st2);
  auto p5 = std::chrono::high_resolution_clock::now();
  if (dev_src_a != nullptr) {
    cudaFree(dev_src_a);
  }
  if (dev_src_b != nullptr) {
    cudaFree(dev_src_b);
  }
  cudaHostUnregister(host_a);
  cudaHostUnregister(host_b);
  auto p6 = std::chrono::high_resolution_clock::now();
  auto cost1 =
      std::chrono::duration_cast<std::chrono::microseconds>(p2 - p1).count();
  auto cost2 =
      std::chrono::duration_cast<std::chrono::microseconds>(p3 - p2).count();
  auto cost3 =
      std::chrono::duration_cast<std::chrono::microseconds>(p4 - p3).count();
  auto cost4 =
      std::chrono::duration_cast<std::chrono::microseconds>(p5 - p4).count();
  auto cost5 =
      std::chrono::duration_cast<std::chrono::microseconds>(p6 - p5).count();
  printf(
      "preprocess_a:%llu,preprocess_b:%llu,preprocess_c:%llu,wait "
      "process:%llu,cudaHostUnregister:%llu\n",
      cost1, cost2, cost3, cost4, cost5);
}

#endif