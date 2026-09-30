// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>

#include <tinyopt/optimizers/optimizer.h>
#include <tinyopt/optimizers/options.h>

#include <tinyopt/optimizers/optimizers.h>
#include <tuple>
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
#if defined(TINYOPT_ENABLE_GRADIENT_DESCENT) || defined(TINYOPT_ENABLE_CONJUGATE_GRADIENT) || \
    defined(TINYOPT_ENABLE_BFGS) || defined(TINYOPT_ENABLE_LBFGS)
  using Gradient_t = std::conditional_t<isDense, Vector<Scalar, Dims>, SparseMatrix<Scalar>>;
#endif

  constexpr bool secondOrderValid = !std::is_invocable_v<Func, const T &, Vector<Scalar, Dims> &>;

  // Check if this is an unconstrained first order problem
#if defined(TINYOPT_ENABLE_GRADIENT_DESCENT) || defined(TINYOPT_ENABLE_CONJUGATE_GRADIENT) || \
    defined(TINYOPT_ENABLE_BFGS) || defined(TINYOPT_ENABLE_LBFGS)
  constexpr bool firstOrderAllowed = !secondOrderValid;
#endif

  switch (options.solver_type) {
    // Second order methods
#if defined(TINYOPT_ENABLE_GAUSS_NEWTON)
    case Options::Solver::GaussNewton:
      if constexpr (secondOrderValid) {
        gn::Optimizer<Hessian_t> optimizer(options);
        return optimizer.Optimize(x, func);
      } else {
        throw std::invalid_argument(
            "Error: GaussNewton can't be used on this gradient only function");
      }
#endif
#if defined(TINYOPT_ENABLE_DOGLEG)
    case Options::Solver::DogLeg:
      if constexpr (secondOrderValid) {
        dl::Optimizer<Hessian_t> optimizer(options);
        return optimizer.Optimize(x, func);
      } else {
        throw std::invalid_argument("Error: DogLeg can't be used on this gradient only function");
      }
#endif
    case Options::Solver::LevenbergMarquardt:
      if constexpr (secondOrderValid) {
        lm::Optimizer<Hessian_t> optimizer(options);
        return optimizer.Optimize(x, func);
      } else {
        throw std::invalid_argument(
            "Error: LevenbergMarquardt can't be used on this gradient only function");
      }
    // First order methods
#if defined(TINYOPT_ENABLE_GRADIENT_DESCENT)
    case Options::Solver::GradientDescent:
      if constexpr (std::is_invocable_v<Func, const T &>) {
        using ReturnType = std::invoke_result_t<Func, T>;
        if constexpr (traits::is_scalar_v<ReturnType>) {
          gd::Optimizer<Gradient_t> optimizer(options);
          return optimizer.Optimize(x, func);
        } else {
          throw std::invalid_argument(
              "Error: cost function must return a scalar for Gradient Descent");
        }
      } else if constexpr (firstOrderAllowed) {
        gd::Optimizer<Gradient_t> optimizer(options);
        return optimizer.Optimize(x, func);
      }
#endif
#if defined(TINYOPT_ENABLE_CONJUGATE_GRADIENT)
    case Options::Solver::ConjugateGradient:
      if constexpr (std::is_invocable_v<Func, const T &>) {
        using ReturnType = std::invoke_result_t<Func, T>;
        if constexpr (traits::is_scalar_v<ReturnType>) {
          cg::Optimizer<Gradient_t> optimizer(options);
          return optimizer.Optimize(x, func);
        } else {
          throw std::invalid_argument(
              "Error: cost function must return a scalar for Conjugate Gradient");
        }
      } else if constexpr (firstOrderAllowed) {
        cg::Optimizer<Gradient_t> optimizer(options);
        return optimizer.Optimize(x, func);
      }
#endif
#if defined(TINYOPT_ENABLE_BFGS)
    case Options::Solver::BFGS:
      if constexpr (std::is_invocable_v<Func, const T &>) {
        using ReturnType = std::invoke_result_t<Func, T>;
        if constexpr (traits::is_scalar_v<ReturnType>) {
          bfgs::Optimizer<Gradient_t> optimizer(options);
          return optimizer.Optimize(x, func);
        } else {
          throw std::invalid_argument("Error: cost function must return a scalar for BFGS");
        }
      } else if constexpr (firstOrderAllowed) {
        bfgs::Optimizer<Gradient_t> optimizer(options);
        return optimizer.Optimize(x, func);
      }
#endif
#if defined(TINYOPT_ENABLE_LBFGS)
    case Options::Solver::LBFGS:
      if constexpr (std::is_invocable_v<Func, const T &>) {
        using ReturnType = std::invoke_result_t<Func, T>;
        if constexpr (traits::is_scalar_v<ReturnType>) {
          lbfgs::Optimizer<Gradient_t> optimizer(options);
          return optimizer.Optimize(x, func);
        } else {
          throw std::invalid_argument("Error: cost function must return a scalar for L-BFGS");
        }
      } else if constexpr (firstOrderAllowed) {
        lbfgs::Optimizer<Gradient_t> optimizer(options);
        return optimizer.Optimize(x, func);
      }
#endif
    default:
      TINYOPT_LOG("❌ Error: Unknown solver type {}", (int)options.solver_type);
      throw std::invalid_argument("Error: Unknown solver type");
  }
}

template <typename T, typename U, typename... Rest, typename Func>
  requires(!std::is_same_v<std::remove_cvref_t<Func>, Options>)
inline Output Optimize(T &x, U &y, Rest &...rest, const Func &func, const Options &options = {}) {
  traits::detail::ParamsPack pack(x, y, rest...);
  auto wrapped = detail::make_packed_adapter<Func, T, U, Rest...>(func);
  return Optimize(pack, wrapped, options);
}

}  // namespace tinyopt
