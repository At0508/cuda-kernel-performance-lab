#include "kernel_reference.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void check(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
}

int main() {
  try {
    const auto product = kernel_lab::matmul_reference({1, 2, 3, 4, 5, 6},
                                                     {7, 8, 9, 10, 11, 12}, 2, 3, 2);
    check(product.size() == 4 && std::abs(product[0] - 58.0f) < 1e-5f &&
          std::abs(product[3] - 154.0f) < 1e-5f, "non-square matrix multiplication");
    check(std::abs(kernel_lab::sum_reference({1, -2, 3, 4}) - 6.0f) < 1e-6f,
          "vector sum");
    const auto probabilities = kernel_lab::row_softmax_reference({1000, 1001, -1000, 0}, 2, 2);
    check(std::abs(probabilities[0] + probabilities[1] - 1.0f) < 1e-6f,
          "softmax row normalization");
    check(std::isfinite(probabilities[0]) && std::isfinite(probabilities[1]),
          "softmax numerical stability");
    const auto odd_product = kernel_lab::matmul_reference(
        {1, 0, -1, 2, 3, 4}, {2, 1, 0, -1, 3, 2}, 2, 3, 2);
    check(odd_product.size() == 4 && std::abs(odd_product[0] - (-1.0f)) < 1e-5f &&
          std::abs(odd_product[3] - 7.0f) < 1e-5f,
          "odd, non-square matrix dimensions");
    bool rejected_empty_softmax = false;
    try {
      (void)kernel_lab::row_softmax_reference({}, 1, 0);
    } catch (const std::invalid_argument&) {
      rejected_empty_softmax = true;
    }
    check(rejected_empty_softmax, "empty softmax shape is rejected");
    std::cout << "PASS: matrix multiplication, sum, stable row softmax\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
