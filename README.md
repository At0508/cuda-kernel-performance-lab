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

NVBench is optional and fetched from NVIDIA's public repository at commit `deb95d3da687fb1e57ba65364210e354aa67198c` (the `python-0.3.0` release). Its RAPIDS-CMake helper is pinned to `edc8773f660e15fa2948324df5dc1a1bb7d9d1cc` (v25.12.00) to avoid changes from a moving dependency branch. It requires CMake 3.30.4+, CUDA Toolkit 12+, and an NVIDIA GPU for measurements. Set the CUDA architecture to match the device you will run on:

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

Save the benchmark output and profiler report with the GPU model, driver, CUDA version, compiler flags, and command. Recorded T4 measurements are linked below.

## Recorded GPU results

On a Tesla T4 in Colab, all three device/reference checks passed: tiled matrix multiplication, one-million-element reduction, and stable row softmax. Their mean CUDA-event timings were **11.994 µs, 1314.532 µs and 13.862 µs**, respectively. These are different operations and shapes, not comparable speedups.

NVBench produced batched timings for all 15 workload configurations. Most cold measurements failed its clock/convergence checks on the shared host; the full warnings and raw results are retained. See the [GPU report, limitations and reproduction commands](docs/results/t4_report_2026-10-15.md). The one-block reduction remains a performance baseline with a clear scaling bottleneck.

## Correctness and limits

CI has separate CPU reference and CUDA compilation jobs. The CUDA job uses a CUDA 12.6 development container and builds both the correctness executable and optional NVBench harness for SM 75 without running GPU code. A successful compilation confirms toolchain compatibility; GPU numerical correctness and performance require an actual NVIDIA device. Runtime checks reject non-finite output errors as failures.

The reduction currently uses one block. This makes its output deterministic and avoids a reset or global atomic in the timed path, but it is a baseline rather than a scalable strategy for very large vectors.

The CPU-only test validates the scalar references and non-square, odd-sized inputs. The CUDA executable independently compares device output with those references. The recorded GPU checks cover the stated inputs, not every possible shape. Do not infer GPU speedups from CPU tests or compare timings without matching workloads. This is an educational kernel study, not a production library.

## Layout

```text
include/       CPU references and CUDA kernel declarations
src/           CPU references, CUDA kernels, correctness and NVBench harnesses
tests/         CPU reference tests
```

## License

MIT. See [LICENSE](LICENSE).
