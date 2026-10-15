#include "cuda_kernels.cuh"

#include <cuda_runtime.h>
#include <math_constants.h>

namespace kernel_lab::cuda {
namespace {
constexpr int kTile = 16;

__global__ void matmul_tiled(const float* a, const float* b, float* c,
                             int rows, int inner, int cols) {
  __shared__ float tile_a[kTile][kTile];
  __shared__ float tile_b[kTile][kTile];
  const int row = blockIdx.y * kTile + threadIdx.y;
  const int col = blockIdx.x * kTile + threadIdx.x;
  float accumulator = 0.0f;
  for (int base = 0; base < inner; base += kTile) {
    tile_a[threadIdx.y][threadIdx.x] =
        (row < rows && base + threadIdx.x < inner)
            ? a[row * inner + base + threadIdx.x] : 0.0f;
    tile_b[threadIdx.y][threadIdx.x] =
        (base + threadIdx.y < inner && col < cols)
            ? b[(base + threadIdx.y) * cols + col] : 0.0f;
    __syncthreads();
    #pragma unroll
    for (int index = 0; index < kTile; ++index)
      accumulator += tile_a[threadIdx.y][index] * tile_b[index][threadIdx.x];
    __syncthreads();
  }
  if (row < rows && col < cols) c[row * cols + col] = accumulator;
}

__global__ void reduce_sum(const float* input, float* output, int count) {
  extern __shared__ float partial[];
  const int tid = threadIdx.x;
  float value = 0.0f;
  for (int index = tid; index < count; index += blockDim.x)
    value += input[index];
  partial[tid] = value;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) partial[tid] += partial[tid + stride];
    __syncthreads();
  }
  if (tid == 0) output[0] = partial[0];
}

__global__ void row_softmax(const float* input, float* output,
                            int rows, int cols) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= rows) return;
  extern __shared__ float scratch[];
  float* maxima = scratch;
  float* sums = scratch + blockDim.x;
  float maximum = -CUDART_INF_F;
  for (int col = tid; col < cols; col += blockDim.x)
    maximum = fmaxf(maximum, input[row * cols + col]);
  maxima[tid] = maximum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) maxima[tid] = fmaxf(maxima[tid], maxima[tid + stride]);
    __syncthreads();
  }
  maximum = maxima[0];
  float denominator = 0.0f;
  for (int col = tid; col < cols; col += blockDim.x)
    denominator += expf(input[row * cols + col] - maximum);
  sums[tid] = denominator;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) sums[tid] += sums[tid + stride];
    __syncthreads();
  }
  denominator = sums[0];
  for (int col = tid; col < cols; col += blockDim.x)
    output[row * cols + col] =
        expf(input[row * cols + col] - maximum) / denominator;
}
}  // namespace

void launch_matmul(const float* a, const float* b, float* c,
                   int rows, int inner, int cols, cudaStream_t stream) {
  const dim3 block(kTile, kTile);
  const dim3 grid((cols + kTile - 1) / kTile, (rows + kTile - 1) / kTile);
  matmul_tiled<<<grid, block, 0, stream>>>(a, b, c, rows, inner, cols);
}

void launch_reduce_sum(const float* input, float* output, int count,
                       cudaStream_t stream) {
  reduce_sum<<<1, 256, 256 * sizeof(float), stream>>>(input, output, count);
}

void launch_row_softmax(const float* input, float* output,
                        int rows, int cols, cudaStream_t stream) {
  constexpr int threads = 256;
  row_softmax<<<rows, threads, 2 * threads * sizeof(float), stream>>>(
      input, output, rows, cols);
}
}  // namespace kernel_lab::cuda
