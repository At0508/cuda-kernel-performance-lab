# CUDA Kernel Performance Lab

Small, measurable CUDA implementations of three common operations: matrix multiplication, reduction, and row-wise softmax. Each operation has a CPU reference so correctness can be checked before looking at timing results.

## What is included

- A tiled matrix multiplication kernel with bounds checks for dimensions that are not multiples of the tile size.
- A one-block reduction kernel for summing a vector.
- A stable row-wise softmax kernel (subtracts each row maximum before exponentiation).
- CPU reference implementations and a CUDA event correctness benchmark.
- Optional NVBench sweeps over matrix, reduction, and softmax workload shapes.
- A CPU-only test path for machines without the CUDA toolkit or an NVIDIA GPU.

## Build and run

### CPU reference tests

Requires a C++17 compiler.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCUDA_BACKEND=OFF
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

On Windows with MinGW, use `-G "MinGW Makefiles"` if CMake does not select the generator automatically.

### CUDA benchmark

Requires CMake 3.24+, a CUDA toolkit supported by the installed NVIDIA driver, and an NVIDIA GPU.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCUDA_BACKEND=ON
cmake --build build --parallel
./build/cuda_benchmark
```

The benchmark emits CSV rows for operation, shape, average CUDA-event latency, maximum absolute error, and pass/fail. It uses fixed seeds and warms up each kernel. Timings cover repeated kernel launches after allocations and host-to-device copies; they do not include those setup costs. Results are specific to the GPU, driver, toolkit, clock state, and workload. Run on an NVIDIA system and save the output alongside those details before comparing configurations.

### NVBench sweeps

NVBench is optional and fetched from NVIDIA's public repository at the `python-0.3.0` tag. It requires CMake 3.30.4+, CUDA Toolkit 12+, and an NVIDIA GPU for measurements. Set the CUDA architecture to match the device you will run on:

```sh
cmake -S . -B build-nvbench -DCMAKE_BUILD_TYPE=Release \
  -DCUDA_BACKEND=ON -DENABLE_NVBENCH=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-nvbench --parallel
./build-nvbench/nvbench_kernels
```

Change `86` to the target GPU's compute capability. NVBench sweeps matrix dimensions, reduction sizes, and softmax shapes; the sweep and correctness executable share the same kernels. Use Nsight Compute to inspect a chosen kernel after benchmarking:

```sh
ncu --set basic -o kernel_profile ./build-nvbench/nvbench_kernels
```

Save the benchmark output and profiler report with the GPU model, driver, CUDA version, compiler flags, and command. No GPU result is included until run on a supported NVIDIA GPU.

## Correctness and limits

The reduction currently uses one block. This makes its output deterministic and avoids a reset or global atomic in the timed path, but it is a baseline rather than a scalable strategy for very large vectors.

The CPU-only test validates the scalar references and non-square, odd-sized inputs. The CUDA executable independently compares device output with those references. CUDA results have not been measured on a GPU in the development environment when no NVIDIA device is available; do not infer GPU speedups from the CPU tests. This is an educational kernel study, not a production library.

## Layout

```text
include/       CPU references and CUDA kernel declarations
src/           CPU references, CUDA kernels, correctness and NVBench harnesses
tests/         CPU reference tests
```

## License

MIT. See [LICENSE](LICENSE).
