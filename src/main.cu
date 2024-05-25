#include "utils.cuh"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "NvInferRuntime.h"
#include "NvInfer.h"
#include <string>
#include <vector>
#include "cublas_v2.h"

__global__ void test(void) {
  int id = threadIdx.x;
  printf("%d,Hello CUDA!\n", id);
}
class Logger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char *msg) { printf("%s \n", msg); }
};
int GetXorBuf(const char *src, char *dst, size_t len, bool is_reverse,
              size_t loops) {
  if (src == nullptr || dst == nullptr || src == dst || len <= 1) {
    std::cerr << "Parameter invalid!";
    return -1;
  }

  if (len > 10e9) {
    std::cerr << "Buffer size " << len << " is too large!";
  }

  if (loops > 10) {
    std::cerr << "Warning, loops " << loops << " is too large!";
  }

  int res_code = 0;

  memcpy(dst, src, len);

  int count = loops * len;
  if (!is_reverse) {
    for (int i = 0; i < count; i++) {
      int idx = i % len;
      int idx_next = (i + 1) % len;
      dst[idx] = dst[idx] ^ dst[idx_next];
    }
  } else {
    for (int i = count - 1; i >= 0; --i) {
      int idx = i % len;
      int idx_next = (i + 1) % len;
      dst[idx] = dst[idx] ^ dst[idx_next];
    }
  }

  return res_code;
}
std::vector<char> readFile(std::string p) {
  auto model_path = std::experimental::filesystem::v1::path(p);
  bool is_di = false;
  {
    auto ext = model_path.extension().string();
    if (ext == ".di") {
      is_di = true;
    }
  }
  std::ifstream file(p, std::ios::binary);
  if (!file.is_open()) {
    return {};
  }
  file.seekg(0, std::ios::end);
  size_t size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<char> res;
  res.resize(size);
  file.read(res.data(), size);
  if (is_di) {
    std::vector<char> t;
    GetXorBuf(res.data(), t.data(), res.size(), true, 3);
    std::swap(res, t);
  }
  return res;
}
size_t modelsize(std::string path) {
  size_t freeMem = 0, totalMem = 0;
  cudaMemGetInfo(&freeMem, &totalMem);
  auto planData = readFile(path);
  Logger loger;
  auto runtime = nvinfer1::createInferRuntime(loger);
  auto engine =
      runtime->deserializeCudaEngine(planData.data(), planData.size());
  int num_layers = engine->getNbLayers();
  auto context = engine->createExecutionContext();
  auto bindings = engine->getNbBindings();
  auto maxBatch = engine->getMaxBatchSize();
  std::cout << maxBatch << std::endl;
  context->setOptimizationProfile(0);
  nvinfer1::Dims input_dims = engine->getBindingDimensions(0);
  nvinfer1::Dims output_dims = engine->getBindingDimensions(1);
  nvinfer1::Dims input_dims_const;
  input_dims_const.nbDims = 4;
  input_dims_const.d[0] = 1;
  input_dims_const.d[1] = 9;
  input_dims_const.d[2] = 192;
  input_dims_const.d[3] = 192;
  context->setBindingDimensions(0, input_dims_const);
  if (!context->allInputDimensionsSpecified()) {
    exit(1);
  }
  void *dev;
  cudaMalloc(&dev, 9 * 192 * 192 * 2);
  //   cudaError_t error = cudaGetLastError();
  //   printf("CUDA error: %s\n", cudaGetErrorString(error));
  //   void* host;
  //   cudaMallocHost(&host, 4 * 256 * 256 * 2);
  //   memset(host, 0, 4 * 256 * 256 * 2);
  //   cudaMemcpy(dev, host, 4 * 256 * 256 * 2, cudaMemcpyHostToDevice);
  void *out;
  cudaMalloc(&out, 192 * 192 * 3 * 2);
  //   error = cudaGetLastError();
  //   printf("CUDA error: %s\n", cudaGetErrorString(error));
  std::vector<void *> temp{dev, out};
  try {
    context->executeV2(temp.data());
  } catch (std::exception e) {
    std::cout << e.what() << std::endl;
    exit(1);
  }
  size_t after = 0;
  cudaMemGetInfo(&after, &totalMem);
  printf("total:%zu,free:%zu \n", totalMem, freeMem);
  printf("total:%zu,free:%zu \n", totalMem, after);
  printf("cost:%zu \n", (freeMem - after) / 1024 / 1024);
  return 0;
}

int main() {
  //   test<<<1, 4>>>();
  //   cudaDeviceSynchronize();
  //   modelsize(
  //       "D:/codes/fpcserver/build-RelWithDebInfo/bin/models/dock/bga/"
  //       "new_user_GXdock_bga_seg_20230315195647/"
  //       "new_user_GXdock_bga_seg_20230315195647_fix_size_norm_dynamic_batch_1_"
  //       "fp16_device_0_NVIDIA_GeForce_RTX_3090.plan");
  using data_type = double;
  cublasHandle_t cublasH = NULL;
  cudaStream_t stream = NULL;

  const int m = 2;
  const int n = 2;
  const int k = 3;
  const int lda = 2;
  const int ldb = 3;
  const int ldc = 2;
  /*
   *   A = | 1.0 | 2.0 | 3.0 |
   *       | 4.0 | 5.0 | 6.0 |
   *
   *   B = | 1.0 | 1.0 |
   *       | 1.0 | 1.0 |
   *       | 1.0 | 1.0 |
   */

  const std::vector<data_type> A = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  const std::vector<data_type> B = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  std::vector<data_type> C(m * n);
  const data_type alpha = 1.0;
  const data_type beta = 0.0;

  data_type *d_A = nullptr;
  data_type *d_B = nullptr;
  data_type *d_C = nullptr;

  cublasOperation_t transa = CUBLAS_OP_N;
  cublasOperation_t transb = CUBLAS_OP_N;
  /* step 1: create cublas handle, bind a stream */
  CUBLAS_CHECK(cublasCreate(&cublasH));

  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUBLAS_CHECK(cublasSetStream(cublasH, stream));

  /* step 2: copy data to device */
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A),
                        sizeof(data_type) * A.size()));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B),
                        sizeof(data_type) * B.size()));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C),
                        sizeof(data_type) * C.size()));

  CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(),
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_B, B.data(), sizeof(data_type) * B.size(),
                             cudaMemcpyHostToDevice, stream));

  /* step 3: compute */
  CUBLAS_CHECK(cublasDgemm(cublasH, transa, transb, m, n, k, &alpha, d_A, lda,
                           d_B, ldb, &beta, d_C, ldc));

  /* step 4: copy data to host */
  CUDA_CHECK(cudaMemcpyAsync(C.data(), d_C, sizeof(data_type) * C.size(),
                             cudaMemcpyDeviceToHost, stream));

  CUDA_CHECK(cudaStreamSynchronize(stream));
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      std::cout << C[i + n * j] << " ";
    }
    std::cout << std::endl;
  }
  return 0;
}
