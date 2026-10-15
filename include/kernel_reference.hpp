#pragma once

#include <cstddef>
#include <vector>

namespace kernel_lab {
std::vector<float> matmul_reference(const std::vector<float>& a,
                                    const std::vector<float>& b,
                                    std::size_t rows, std::size_t inner,
                                    std::size_t cols);
float sum_reference(const std::vector<float>& values);
std::vector<float> row_softmax_reference(const std::vector<float>& input,
                                         std::size_t rows,
                                         std::size_t cols);
}
