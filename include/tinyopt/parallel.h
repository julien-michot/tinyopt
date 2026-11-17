// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <type_traits>

#if defined(TINYOPT_USE_OPENMP)
#include <Eigen/Core>
#include <omp.h>
#endif

namespace tinyopt::parallel {

[[nodiscard]] inline int MaxThreads() noexcept {
#if defined(TINYOPT_USE_OPENMP) && !defined(TINYOPT_ENFORCE_NO_DYNAMIC_ALLOCATIONS)
  return omp_get_max_threads();
#else
  return 1;
#endif
}

template <typename Function>
inline void ParallelFor(std::size_t count, std::size_t min_count, std::uint16_t num_threads,
                        Function&& function) {
#if defined(TINYOPT_USE_OPENMP) && !defined(TINYOPT_ENFORCE_NO_DYNAMIC_ALLOCATIONS)
  const int threads = num_threads == 0 ? omp_get_max_threads() : static_cast<int>(num_threads);
  if (threads > 1 && count >= min_count) {
#pragma omp parallel for schedule(static) num_threads(threads)
    for (std::int64_t index = 0; index < static_cast<std::int64_t>(count); ++index) {
      if constexpr (std::is_invocable_v<Function&, std::size_t, int>) {
        std::invoke(function, static_cast<std::size_t>(index), omp_get_thread_num());
      } else {
        std::invoke(function, static_cast<std::size_t>(index));
      }
    }
    return;
  }
#else
  static_cast<void>(min_count);
  static_cast<void>(num_threads);
#endif
  for (std::size_t index = 0; index < count; ++index) {
    if constexpr (std::is_invocable_v<Function&, std::size_t, int>) {
      std::invoke(function, index, 0);
    } else {
      std::invoke(function, index);
    }
  }
}

[[nodiscard]] inline int ThreadCount(std::uint16_t num_threads) noexcept {
#if defined(TINYOPT_USE_OPENMP) && !defined(TINYOPT_ENFORCE_NO_DYNAMIC_ALLOCATIONS)
  return num_threads == 0 ? omp_get_max_threads() : static_cast<int>(num_threads);
#else
  static_cast<void>(num_threads);
  return 1;
#endif
}

class ScopedEigenThreads {
 public:
  explicit ScopedEigenThreads(std::uint16_t num_threads)
#if defined(TINYOPT_USE_OPENMP) && !defined(TINYOPT_ENFORCE_NO_DYNAMIC_ALLOCATIONS)
      : lock_(Mutex()), previous_(Eigen::nbThreads())
#endif
  {
#if defined(TINYOPT_USE_OPENMP) && !defined(TINYOPT_ENFORCE_NO_DYNAMIC_ALLOCATIONS)
    Eigen::setNbThreads(ThreadCount(num_threads));
#else
    static_cast<void>(num_threads);
#endif
  }

  ~ScopedEigenThreads() {
#if defined(TINYOPT_USE_OPENMP) && !defined(TINYOPT_ENFORCE_NO_DYNAMIC_ALLOCATIONS)
    Eigen::setNbThreads(previous_);
#endif
  }

  ScopedEigenThreads(const ScopedEigenThreads&) = delete;
  ScopedEigenThreads& operator=(const ScopedEigenThreads&) = delete;

 private:
#if defined(TINYOPT_USE_OPENMP) && !defined(TINYOPT_ENFORCE_NO_DYNAMIC_ALLOCATIONS)
  static std::recursive_mutex& Mutex() {
    static std::recursive_mutex mutex;
    return mutex;
  }

  std::unique_lock<std::recursive_mutex> lock_;
  int previous_;
#endif
};

template <typename Function>
[[nodiscard]] inline bool ParallelForAnd(std::size_t count, std::size_t min_count,
                                         std::uint16_t num_threads, Function&& function) {
  bool result = true;
#if defined(TINYOPT_USE_OPENMP) && !defined(TINYOPT_ENFORCE_NO_DYNAMIC_ALLOCATIONS)
  const int threads = num_threads == 0 ? omp_get_max_threads() : static_cast<int>(num_threads);
  if (threads > 1 && count >= min_count) {
#pragma omp parallel for schedule(static) num_threads(threads) reduction(&& : result)
    for (std::int64_t index = 0; index < static_cast<std::int64_t>(count); ++index) {
      result = std::invoke(function, static_cast<std::size_t>(index)) && result;
    }
    return result;
  }
#else
  static_cast<void>(min_count);
  static_cast<void>(num_threads);
#endif
  for (std::size_t index = 0; index < count; ++index) {
    result = std::invoke(function, index) && result;
  }
  return result;
}

}  // namespace tinyopt::parallel
