// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <iostream>
#include <sstream>

#include <tinyopt/traits.h>

namespace tinyopt::benchmark {

/// Struct that collects Convergence rate and number of iterations and print at the end.
template <typename T>
struct StatCounter {
  StatCounter() = default;
  ~StatCounter() {
    std::ostringstream oss;
    oss << "'";
    if constexpr (traits::is_matrix_or_array_v<T>)
      oss << "Mat" << T::RowsAtCompileTime << "x" << T::ColsAtCompileTime;
    else
      oss << typeid(T).name();
    oss << "' mean [iters: " << sum_total_iters / (float)NumSamples()
        << ", success:" << 100.0f * ConvRatio() << "%] n:" << NumSamples() << "\n";
    std::cout << oss.str();
  }

  void AddConv(bool converged) {
    if (converged)
      num_converged++;
    else
      num_not_converged++;
  }
  void AddFinalIters(int n) { sum_total_iters += n; }

  std::size_t NumSamples() const { return num_converged + num_not_converged; }
  float ConvRatio() const { return num_converged / (NumSamples() + 1e-6); }

  std::size_t num_converged = 0;
  std::size_t num_not_converged = 0;
  std::size_t sum_total_iters = 0;
};

}  // namespace tinyopt::benchmark