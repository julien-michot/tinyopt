// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <sycl/sycl.hpp>

#include <tinyopt/optimize.h>

namespace tinyopt {

/// One device-computed factor contribution for a fixed-size dense parameter vector.
template <typename Scalar, int Dims>
struct SyclFactorContribution {
  Scalar cost = Scalar(0);
  int num_residuals = 0;
  std::array<Scalar, Dims> gradient{};
  std::array<Scalar, Dims * Dims> hessian{};
};

namespace detail {

template <typename X_t, typename FactorFunction>
class SyclFactorAccumulator {
 public:
  using Scalar = typename traits::params_trait<X_t>::Scalar;
  static constexpr int Dims = traits::params_trait<X_t>::Dims;
  using Contribution = SyclFactorContribution<Scalar, Dims>;
  static constexpr std::size_t OutputCount = 2 + Dims + Dims * Dims;
  static constexpr std::size_t ReductionWorkGroupSize = 128;

  SyclFactorAccumulator(sycl::queue& queue, Index num_factors, FactorFunction factor)
      : queue_(queue),
        num_factors_(static_cast<std::size_t>(num_factors)),
        factor_(std::move(factor)),
        contributions_(sycl::range<1>(num_factors_)),
        reduced_(sycl::range<1>(OutputCount)) {}

  Cost operator()(const X_t& value, auto& gradient, auto& hessian) const {
    std::array<Scalar, Dims> parameters{};
    for (int i = 0; i < Dims; ++i) parameters[static_cast<std::size_t>(i)] = value[i];
    const FactorFunction factor = factor_;
    const std::size_t factor_count = num_factors_;

    queue_.submit([&](sycl::handler& handler) {
      auto output = contributions_.template get_access<sycl::access::mode::write>(handler);
      handler.parallel_for(sycl::range<1>(factor_count), [=](sycl::id<1> id) {
        output[id] = factor(static_cast<Index>(id[0]), parameters);
      });
    });

    queue_.submit([&](sycl::handler& handler) {
      auto factors = contributions_.template get_access<sycl::access::mode::read>(handler);
      auto output = reduced_.template get_access<sycl::access::mode::write>(handler);
      sycl::local_accessor<Scalar, 1> partials(sycl::range<1>(ReductionWorkGroupSize), handler);
      handler.parallel_for(
          sycl::nd_range<2>(sycl::range<2>(OutputCount, ReductionWorkGroupSize),
                            sycl::range<2>(1, ReductionWorkGroupSize)),
          [=](sycl::nd_item<2> item) {
        const std::size_t component = item.get_group(0);
        const std::size_t lane = item.get_local_id(1);
        Scalar total = Scalar(0);
        for (std::size_t factor_index = lane; factor_index < factor_count;
             factor_index += ReductionWorkGroupSize) {
          if (component == 0) {
            total += factors[factor_index].cost;
          } else if (component == 1) {
            total += static_cast<Scalar>(factors[factor_index].num_residuals);
          } else if (component < static_cast<std::size_t>(2 + Dims)) {
            total += factors[factor_index].gradient[component - 2];
          } else {
            total += factors[factor_index].hessian[component - (2 + Dims)];
          }
        }
        partials[lane] = total;
        item.barrier(sycl::access::fence_space::local_space);
        for (std::size_t offset = ReductionWorkGroupSize / 2; offset > 0; offset /= 2) {
          if (lane < offset) partials[lane] += partials[lane + offset];
          item.barrier(sycl::access::fence_space::local_space);
        }
        if (lane == 0) output[component] = partials[0];
      });
    });

    queue_.wait_and_throw();
    const auto output = reduced_.template get_host_access();
    if constexpr (!traits::is_nullptr_v<decltype(gradient)>) {
      for (int i = 0; i < Dims; ++i)
        gradient[i] += output[static_cast<std::size_t>(2 + i)];
    }
    if constexpr (!traits::is_nullptr_v<decltype(hessian)>) {
      for (int row = 0; row < Dims; ++row) {
        for (int col = 0; col < Dims; ++col) {
          const std::size_t index =
              static_cast<std::size_t>(2 + Dims + row * Dims + col);
          hessian(row, col) += output[index];
        }
      }
    }
    return Cost(static_cast<double>(output[0]), static_cast<int>(output[1]));
  }

 private:
  sycl::queue& queue_;
  std::size_t num_factors_;
  FactorFunction factor_;
  mutable sycl::buffer<Contribution, 1> contributions_;
  mutable sycl::buffer<Scalar, 1> reduced_;
};

}  // namespace detail

/// Optimize factors evaluated on a SYCL device for fixed-size dense Eigen column vectors.
///
/// The factor must be device-copyable and return its cost, residual count, gradient, and Hessian
/// contribution in a SyclFactorContribution. Existing host callbacks remain available through the
/// non-SYCL OptimizeFactors() overload in optimize_factors.h.
template <typename X_t, typename FactorFunction>
inline Summary OptimizeFactors(sycl::queue& queue, X_t& x, Index num_factors,
                               FactorFunction factor, const Options& options = {}) {
  using Scalar = typename traits::params_trait<X_t>::Scalar;
  constexpr int Dims = traits::params_trait<X_t>::Dims;
  static_assert(Dims > 0, "SYCL factor optimization requires fixed-size parameters");
  static_assert(X_t::ColsAtCompileTime == 1, "SYCL factor optimization requires a column vector");
  static_assert(std::is_floating_point_v<Scalar>, "SYCL factor optimization requires float/double");
  if (num_factors <= 0) throw std::invalid_argument("num_factors must be positive");

  detail::SyclFactorAccumulator<X_t, FactorFunction> accumulator(queue, num_factors,
                                                                 std::move(factor));
  return Optimize(x, accumulator, options);
}

}  // namespace tinyopt
