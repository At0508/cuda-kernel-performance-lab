#include "cuda_kernels.cuh"

#include <nvbench/nvbench.cuh>

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check_cuda(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(status));
}

template <class T>
T* upload(const std::vector<T>& values) {
  T* device_values = nullptr;
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_values), values.size() * sizeof(T)));
  try {
    check_cuda(cudaMemcpy(device_values, values.data(), values.size() * sizeof(T),
                          cudaMemcpyHostToDevice));
  } catch (...) {
    cudaFree(device_values);
    throw;
  }
  return device_values;
}

void matmul_benchmark(nvbench::state& state) {
  const int rows = static_cast<int>(state.get_int64("Rows"));
  const int inner = static_cast<int>(state.get_int64("Inner"));
  const int cols = static_cast<int>(state.get_int64("Columns"));
  std::vector<float> host_a(static_cast<std::size_t>(rows) * inner, 0.25f);
  std::vector<float> host_b(static_cast<std::size_t>(inner) * cols, -0.5f);
  float* a = upload(host_a);
  float* b = upload(host_b);
  float* c = nullptr;
  try {
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&c),
                          static_cast<std::size_t>(rows) * cols * sizeof(float)));
    state.add_element_count(static_cast<std::size_t>(rows) * cols, "OutputElements");
    state.exec([=](nvbench::launch& launch) {
      kernel_lab::cuda::launch_matmul(a, b, c, rows, inner, cols, launch.get_stream());
    });
  } catch (...) {
    cudaFree(a); cudaFree(b); cudaFree(c);
    throw;
  }
  cudaFree(a); cudaFree(b); cudaFree(c);
}
NVBENCH_BENCH(matmul_benchmark)
  .add_int64_axis("Rows", {64, 128})
  .add_int64_axis("Inner", {64, 128})
  .add_int64_axis("Columns", {64, 128});

void reduction_benchmark(nvbench::state& state) {
  const int count = static_cast<int>(state.get_int64("NumInputs"));
  std::vector<float> host_input(static_cast<std::size_t>(count), 1.0f);
  float* input = upload(host_input);
  float* output = nullptr;
  try {
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&output), sizeof(float)));
    state.add_element_count(static_cast<std::size_t>(count), "NumInputs");
    state.exec([=](nvbench::launch& launch) {
      kernel_lab::cuda::launch_reduce_sum(input, output, count, launch.get_stream());
    });
  } catch (...) {
    cudaFree(input); cudaFree(output);
    throw;
  }
  cudaFree(input); cudaFree(output);
}
NVBENCH_BENCH(reduction_benchmark)
  .add_int64_axis("NumInputs", {4096, 65536, 1048576});

void softmax_benchmark(nvbench::state& state) {
  const int rows = static_cast<int>(state.get_int64("Rows"));
  const int cols = static_cast<int>(state.get_int64("Columns"));
  std::vector<float> host_input(static_cast<std::size_t>(rows) * cols, 0.125f);
  float* input = upload(host_input);
  float* output = nullptr;
  try {
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&output),
                          static_cast<std::size_t>(rows) * cols * sizeof(float)));
    state.add_element_count(static_cast<std::size_t>(rows) * cols, "Elements");
    state.exec([=](nvbench::launch& launch) {
      kernel_lab::cuda::launch_row_softmax(input, output, rows, cols,
                                           launch.get_stream());
    });
  } catch (...) {
    cudaFree(input); cudaFree(output);
    throw;
  }
  cudaFree(input); cudaFree(output);
}
NVBENCH_BENCH(softmax_benchmark)
  .add_int64_axis("Rows", {64, 257})
  .add_int64_axis("Columns", {128, 513});
}  // namespace
