// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <exception>
#include <stdexcept>
#include <vector>

#include <tinyopt/optimize.h>
#include <tinyopt/parallel.h>

namespace tinyopt {

/// Optimize independently accumulated factors using per-thread dense normal-equation buffers.
template <typename X_t, typename FactorFunction>
inline Summary OptimizeFactors(X_t& x, Index num_factors, const FactorFunction& factor,
                               const Options& options = {}) {
  if (num_factors <= 0) throw std::invalid_argument("num_factors must be positive");

  using Scalar = typename traits::params_trait<X_t>::Scalar;
  constexpr Index Dims = traits::params_trait<X_t>::Dims;
  using Gradient = Vector<Scalar, Dims>;
  using Hessian = Matrix<Scalar, Dims, Dims>;

  class FactorAccumulator {
   public:
    FactorAccumulator(const FactorFunction& factor_function, Index factor_count, Index dims,
                      const Options& options)
        : factor_(factor_function),
          factor_count_(factor_count),
          num_threads_(options.num_threads),
          worker_count_(factor_count >= 16 ? parallel::ThreadCount(options.num_threads) : 1) {
      if (worker_count_ <= 1) return;
      gradients_.resize(static_cast<std::size_t>(worker_count_));
      hessians_.resize(static_cast<std::size_t>(worker_count_));
      costs_.resize(static_cast<std::size_t>(worker_count_));
      errors_.resize(static_cast<std::size_t>(worker_count_));
      for (int i = 0; i < worker_count_; ++i) {
        gradients_[static_cast<std::size_t>(i)].resize(dims);
        hessians_[static_cast<std::size_t>(i)].resize(dims, dims);
      }
    }

    Cost operator()(const X_t& value, Gradient& gradient, Hessian& hessian) const {
      if (worker_count_ <= 1) {
        Cost total;
        for (Index i = 0; i < factor_count_; ++i) total += factor_(i, value, gradient, hessian);
        return total;
      }

      for (int i = 0; i < worker_count_; ++i) {
        gradients_[static_cast<std::size_t>(i)].setZero();
        hessians_[static_cast<std::size_t>(i)].setZero();
        costs_[static_cast<std::size_t>(i)] = Cost{};
        errors_[static_cast<std::size_t>(i)] = nullptr;
      }
      parallel::ParallelFor(static_cast<std::size_t>(factor_count_), 16, num_threads_,
                            [&](std::size_t i, int thread_id) {
                              const auto thread = static_cast<std::size_t>(thread_id);
                              if (errors_[thread] != nullptr) return;
                              try {
                                costs_[thread] += factor_(static_cast<Index>(i), value,
                                                          gradients_[thread], hessians_[thread]);
                              } catch (...) {
                                errors_[thread] = std::current_exception();
                              }
                            });

      for (const auto& error : errors_) {
        if (error != nullptr) std::rethrow_exception(error);
      }

      Cost total;
      for (int i = 0; i < worker_count_; ++i) {
        const auto thread = static_cast<std::size_t>(i);
        gradient += gradients_[thread];
        hessian += hessians_[thread];
        total += costs_[thread];
      }
      return total;
    }

   private:
    const FactorFunction& factor_;
    Index factor_count_;
    std::uint16_t num_threads_;
    int worker_count_;
    mutable std::vector<Gradient> gradients_;
    mutable std::vector<Hessian> hessians_;
    mutable std::vector<Cost> costs_;
    mutable std::vector<std::exception_ptr> errors_;
  };

  FactorAccumulator accumulator(factor, num_factors, traits::DynDims(x), options);
  return Optimize(x, accumulator, options);
}

}  // namespace tinyopt
