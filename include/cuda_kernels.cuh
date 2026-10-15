#pragma once

#include <cuda_runtime_api.h>

namespace kernel_lab::cuda {
void launch_matmul(const float* a, const float* b, float* c,
                   int rows, int inner, int cols, cudaStream_t stream = nullptr);
void launch_reduce_sum(const float* input, float* output, int count,
                       cudaStream_t stream = nullptr);
void launch_row_softmax(const float* input, float* output,
                        int rows, int cols, cudaStream_t stream = nullptr);
}  // namespace kernel_lab::cuda
