// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <cstdio>
#include <utility>

namespace tinyopt::benchmark {

template <typename Function>
auto MeasureMilliseconds(Function&& function) {
  const auto start = std::chrono::steady_clock::now();
  auto result = std::forward<Function>(function)();
  const auto elapsed = std::chrono::steady_clock::now() - start;
  return std::pair{std::move(result), std::chrono::duration<double, std::milli>(elapsed).count()};
}

inline void PrintThreadComparison(const char* backend, double one_thread_ms, double eight_threads_ms,
                                  double one_thread_cost, double eight_threads_cost) {
  std::printf("MT_RESULT backend=%s one_thread_ms=%.3f eight_threads_ms=%.3f speedup=%.3f "
              "one_thread_cost=%.12g eight_threads_cost=%.12g\n",
              backend, one_thread_ms, eight_threads_ms,
              eight_threads_ms > 0.0 ? one_thread_ms / eight_threads_ms : 0.0, one_thread_cost,
              eight_threads_cost);
}

}  // namespace tinyopt::benchmark
