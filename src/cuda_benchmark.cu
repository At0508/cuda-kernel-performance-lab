#include "kernel_reference.hpp"
#include "cuda_kernels.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call) do { \
  cudaError_t e = (call); \
  if (e != cudaSuccess) throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(e)); \
} while (0)

namespace {
template <typename F>
float timed_ms(F&& launch, int warmup = 20, int repetitions = 100) {
  for (int i = 0; i < warmup; ++i) launch();
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < repetitions; ++i) launch();
  CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
  CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
  return elapsed * 1000.0f / repetitions;
}

float max_error(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) return INFINITY;
  float e = 0.0f;
  for (std::size_t i = 0; i < a.size(); ++i) e = std::max(e, std::abs(a[i] - b[i]));
  return e;
}

template <typename T> class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t n) : count_(n) {
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), n * sizeof(T)));
  }
  ~DeviceBuffer() { if (ptr_) cudaFree(ptr_); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  T* get() { return ptr_; }
  void upload(const std::vector<T>& src) { CUDA_CHECK(cudaMemcpy(ptr_, src.data(), count_ * sizeof(T), cudaMemcpyHostToDevice)); }
  std::vector<T> download() const {
    std::vector<T> out(count_); CUDA_CHECK(cudaMemcpy(out.data(), ptr_, count_ * sizeof(T), cudaMemcpyDeviceToHost)); return out;
  }
 private:
  T* ptr_ = nullptr;
  std::size_t count_;
};

void emit(const char* op, const char* shape, float us, float error, float tolerance) {
  std::cout << op << ',' << shape << ',' << std::fixed << std::setprecision(3) << us
            << ',' << std::scientific << error << ',' << (error <= tolerance ? "PASS" : "FAIL") << '\n';
  if (error > tolerance) throw std::runtime_error(std::string(op) + " failed CPU reference check");
}
}

int main() {
  try {
    int device = 0; CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{}; CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::cout << "# device," << prop.name << "\n# warmup,20\n# repetitions,100\n";
    std::cout << "operation,shape,mean_kernel_us,max_abs_error,check\n";
    std::mt19937 rng(42); std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    const int m = 97, k = 73, n = 111;
    std::vector<float> a(m*k), b(k*n);
    for (auto& v : a) v = dist(rng); for (auto& v : b) v = dist(rng);
    DeviceBuffer<float> da(a.size()), db(b.size()), dc(m*n);
    da.upload(a); db.upload(b);
    const float ms = timed_ms([&] {
      kernel_lab::cuda::launch_matmul(da.get(), db.get(), dc.get(), m, k, n);
    });
    emit("matmul", "97x73x111", ms, max_error(dc.download(), kernel_lab::matmul_reference(a,b,m,k,n)), 2e-4f);

    std::vector<float> x(1 << 20); for (auto& v : x) v = dist(rng);
    DeviceBuffer<float> dx(x.size()), dr(1); dx.upload(x);
    // A single block keeps the output deterministic and avoids including a
    // result-buffer reset or a second reduction launch in the timed region.
    const float reduce_ms = timed_ms([&] {
      kernel_lab::cuda::launch_reduce_sum(dx.get(), dr.get(), static_cast<int>(x.size()));
    });
    const float result = dr.download()[0];
    emit("sum", "1048576", reduce_ms, std::abs(result-kernel_lab::sum_reference(x)), 0.08f);

    const int rows=257, cols=513;
    std::vector<float> logits(rows*cols); for (auto& v : logits) v=dist(rng)*12.0f;
    DeviceBuffer<float> dl(logits.size()), dy(logits.size()); dl.upload(logits);
    const float softmax_ms=timed_ms([&] {
      kernel_lab::cuda::launch_row_softmax(dl.get(), dy.get(), rows, cols);
    });
    emit("row_softmax", "257x513", softmax_ms, max_error(dy.download(),kernel_lab::row_softmax_reference(logits,rows,cols)), 2e-5f);
    return 0;
  } catch (const std::exception& e) { std::cerr << "ERROR: " << e.what() << '\n'; return 1; }
}
