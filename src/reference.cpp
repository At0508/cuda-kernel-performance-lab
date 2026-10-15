#include "kernel_reference.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace kernel_lab {
namespace {
void require_size(std::size_t actual, std::size_t expected, const char* name) {
  if (actual != expected) throw std::invalid_argument(std::string(name) + " has an unexpected size");
}
std::size_t checked_product(std::size_t left, std::size_t right, const char* name) {
  if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right)
    throw std::invalid_argument(std::string(name) + " dimensions overflow");
  return left * right;
}
}

std::vector<float> matmul_reference(const std::vector<float>& a,
                                    const std::vector<float>& b,
                                    std::size_t rows, std::size_t inner,
                                    std::size_t cols) {
  require_size(a.size(), checked_product(rows, inner, "A"), "A");
  require_size(b.size(), checked_product(inner, cols, "B"), "B");
  std::vector<float> out(checked_product(rows, cols, "output"), 0.0f);
  for (std::size_t i = 0; i < rows; ++i)
    for (std::size_t j = 0; j < cols; ++j) {
      double acc = 0.0;
      for (std::size_t k = 0; k < inner; ++k)
        acc += static_cast<double>(a[i * inner + k]) * b[k * cols + j];
      out[i * cols + j] = static_cast<float>(acc);
    }
  return out;
}

float sum_reference(const std::vector<float>& values) {
  double acc = 0.0;
  for (float value : values) acc += value;
  return static_cast<float>(acc);
}

std::vector<float> row_softmax_reference(const std::vector<float>& input,
                                         std::size_t rows,
                                         std::size_t cols) {
  if (rows == 0 || cols == 0)
    throw std::invalid_argument("softmax dimensions must be nonzero");
  require_size(input.size(), checked_product(rows, cols, "input"), "input");
  std::vector<float> out(input.size());
  for (std::size_t r = 0; r < rows; ++r) {
    const auto begin = input.begin() + r * cols;
    const float maximum = *std::max_element(begin, begin + cols);
    double denominator = 0.0;
    for (std::size_t c = 0; c < cols; ++c) {
      const float value = std::exp(input[r * cols + c] - maximum);
      out[r * cols + c] = value;
      denominator += value;
    }
    for (std::size_t c = 0; c < cols; ++c)
      out[r * cols + c] = static_cast<float>(out[r * cols + c] / denominator);
  }
  return out;
}
}
