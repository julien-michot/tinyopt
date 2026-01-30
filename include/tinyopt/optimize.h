// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tinyopt/optimizers/optimizer.h>
#include <tinyopt/optimizers/options.h>

#include <tinyopt/optimizers/optimizers.h>
#include "tinyopt/log.h"

namespace tinyopt {

/// Simplest interface to optimize `x` and minimize residuals (loss function).
/// Internally call the optimizer and run the optimization.
template <typename T, typename Func>
inline Output Optimize(T &x, const Func &func, const Options &options = {}) {
  // Detect Scalar, supporting at most one nesting level
  using Scalar = std::conditional_t<
      std::is_scalar_v<typename traits::params_trait<T>::Scalar>,
      typename traits::params_trait<T>::Scalar,
      typename traits::params_trait<typename traits::params_trait<T>::Scalar>::Scalar>;
  static_assert(std::is_scalar_v<Scalar>);
  constexpr Index Dims = traits::params_trait<T>::Dims;

  // Detect Hessian Type, if it's dense or sparse
  constexpr bool isDense =
      std::is_invocable_v<Func, const T &> ||
      std::is_invocable_v<Func, const T &, Vector<Scalar, Dims> &> ||
      std::is_invocable_v<Func, const T &, Vector<Scalar, Dims> &, Matrix<Scalar, Dims, Dims> &>;

  using Hessian_t = std::conditional_t<isDense, Matrix<Scalar, Dims, Dims>, SparseMatrix<Scalar>>;
  using Gradient_t = std::conditional_t<isDense, Vector<Scalar, Dims>, SparseMatrix<Scalar>>;

  constexpr bool secondOrderValid = !std::is_invocable_v<Func, const T &, Vector<Scalar, Dims> &>;

  // Check if this is an unconstrained first order problem
  constexpr bool firstOrderAllowed = !secondOrderValid;

  switch (options.solver_type) {
    // Second order methods
    case Options::Solver::GaussNewton:
      if constexpr (secondOrderValid) {
        gn::Optimizer<Hessian_t> optimizer(options);
        return optimizer(x, func);
      } else {
        throw std::invalid_argument(
            "Error: GaussNewton can't be used on this gradient only function");
      }
    case Options::Solver::LevenbergMarquardt:
      if constexpr (secondOrderValid) {
        lm::Optimizer<Hessian_t> optimizer(options);
        return optimizer(x, func);
      } else {
        throw std::invalid_argument(
            "Error: LevenbergMarquardt can't be used on this gradient only function");
      }
    // First order methods
    case Options::Solver::GradientDescent:
      if constexpr (std::is_invocable_v<Func, const T &>) {
        using ReturnType = std::invoke_result_t<Func, T>;
        if constexpr (traits::is_scalar_v<ReturnType>) {
          gd::Optimizer<Gradient_t> optimizer(options);
          return optimizer(x, func);
        } else {
          throw std::invalid_argument(
              "Error: cost function must return a scalar for Gradient Descent");
        }
      } else if constexpr (firstOrderAllowed) {
        gd::Optimizer<Gradient_t> optimizer(options);
        return optimizer(x, func);
      }
    default:
      TINYOPT_LOG("❌ Error: Unknown solver type {}", (int)options.solver_type);
      throw std::invalid_argument("Error: Unknown solver type");
  }
}

}  // namespace tinyopt
