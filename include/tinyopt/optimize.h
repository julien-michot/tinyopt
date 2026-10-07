// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdexcept>
#include <type_traits>

#include <tinyopt/optimizers/optimizer1.h>
#include <tinyopt/optimizers/optimizer2.h>
#include <tinyopt/optimizers/bfgs.h>
#include <tinyopt/optimizers/lbfgs.h>
#include <tinyopt/optimizers/cg.h>
#include <tinyopt/optimizers/dl.h>
#include <tinyopt/optimizers/gd.h>
#include <tinyopt/optimizers/gn.h>
#include <tinyopt/optimizers/lm.h>
#include <tinyopt/optimizers/options.h>
#include <tinyopt/log.h>

namespace tinyopt {

namespace detail {

template <typename T, typename Func, typename Gradient>
Summary RunFirstOrder(T &x, const Func &func, const Options &options) {
  constexpr bool scalar_cost = std::is_invocable_v<const Func &, const T &>;
  constexpr bool gradient_accumulation =
      std::is_invocable_v<const Func &, const T &, Gradient &>;

  if constexpr (scalar_cost) {
    using Return = std::invoke_result_t<const Func &, const T &>;
    if constexpr (traits::is_scalar_v<Return>) {
      switch (options.solver_type) {
        case Options::Solver::ConjugateGradient:
          return cg::Optimizer<Gradient>(options).Optimize(x, func);
        case Options::Solver::BFGS:
          return bfgs::Optimizer<Gradient>(options).Optimize(x, func);
        case Options::Solver::LBFGS:
          return lbfgs::Optimizer<Gradient>(options).Optimize(x, func);
        case Options::Solver::GradientDescent:
          return gd::Optimizer<Gradient>(options).Optimize(x, func);
        default:
          throw std::invalid_argument("Invalid solver for first-order optimization");
      }
    } else {
      throw std::invalid_argument(
          "First-order optimization requires a scalar cost or a gradient accumulator");
    }
  } else if constexpr (gradient_accumulation) {
    switch (options.solver_type) {
      case Options::Solver::ConjugateGradient:
        return cg::Optimizer<Gradient>(options).Optimize(x, func);
      case Options::Solver::BFGS:
        return bfgs::Optimizer<Gradient>(options).Optimize(x, func);
      case Options::Solver::LBFGS:
        return lbfgs::Optimizer<Gradient>(options).Optimize(x, func);
      case Options::Solver::GradientDescent:
        return gd::Optimizer<Gradient>(options).Optimize(x, func);
      default:
        throw std::invalid_argument("Invalid solver for first-order optimization");
    }
  } else {
    throw std::invalid_argument("Invalid function for first-order optimization");
  }
}

template <typename T, typename Func, typename Hessian, typename Gradient>
Summary RunSecondOrder(T &x, const Func &func, const Options &options) {
  constexpr bool second_order_valid =
      !std::is_invocable_v<const Func &, const T &, Gradient &>;
  if constexpr (second_order_valid) {
    switch (options.solver_type) {
      case Options::Solver::LevenbergMarquardt:
        return lm::Optimizer<Hessian>(options).Optimize(x, func);
      case Options::Solver::GaussNewton:
        return gn::Optimizer<Hessian>(options).Optimize(x, func);
      case Options::Solver::DogLeg:
        return dl::Optimizer<Hessian>(options).Optimize(x, func);
      default:
        throw std::invalid_argument("Invalid solver for second-order optimization");
    }
  } else {
    throw std::invalid_argument(
        "Second-order optimization cannot use a gradient-only function");
  }
}

}  // namespace detail

/// Optimize `x` using the selected first- or second-order algorithm.
template <typename T, typename Func>
inline Summary Optimize(T &x, const Func &func, const Options &options = {}) {
  using Scalar = std::conditional_t<
      std::is_scalar_v<typename traits::params_trait<T>::Scalar>,
      typename traits::params_trait<T>::Scalar,
      typename traits::params_trait<typename traits::params_trait<T>::Scalar>::Scalar>;
  static_assert(std::is_scalar_v<Scalar>);
  constexpr Index Dims = traits::params_trait<T>::Dims;
  constexpr bool dense =
      std::is_invocable_v<const Func &, const T &> ||
      std::is_invocable_v<const Func &, const T &, Vector<Scalar, Dims> &> ||
      std::is_invocable_v<const Func &, const T &, Vector<Scalar, Dims> &,
                          Matrix<Scalar, Dims, Dims> &>;
  using Hessian = std::conditional_t<dense, Matrix<Scalar, Dims, Dims>, SparseMatrix<Scalar>>;
  using Gradient = std::conditional_t<dense, Vector<Scalar, Dims>, SparseMatrix<Scalar>>;

  constexpr bool has_gradient_accumulator =
      std::is_invocable_v<const Func &, const T &, Vector<Scalar, Dims> &>;
  constexpr bool has_hessian_accumulator =
      std::is_invocable_v<const Func &, const T &, Vector<Scalar, Dims> &,
                          Matrix<Scalar, Dims, Dims> &>;
  if constexpr (has_gradient_accumulator && !has_hessian_accumulator) {
    return detail::RunFirstOrder<T, Func, Gradient>(x, func, options);
  } else if constexpr (has_hessian_accumulator && !has_gradient_accumulator) {
    return detail::RunSecondOrder<T, Func, Hessian, Vector<Scalar, Dims>>(x, func, options);
  } else {
    switch (options.solver_type) {
      case Options::Solver::LevenbergMarquardt:
      case Options::Solver::GaussNewton:
      case Options::Solver::DogLeg:
        return detail::RunSecondOrder<T, Func, Hessian, Vector<Scalar, Dims>>(x, func, options);
      case Options::Solver::GradientDescent:
      case Options::Solver::ConjugateGradient:
      case Options::Solver::BFGS:
      case Options::Solver::LBFGS:
        return detail::RunFirstOrder<T, Func, Gradient>(x, func, options);
      default:
        TINYOPT_LOG("❌ Error: Unknown solver type {}", static_cast<int>(options.solver_type));
        throw std::invalid_argument("Error: Unknown solver type");
    }
  }
}

template <typename T, typename U, typename... Rest, typename Func>
  requires(!std::is_same_v<std::remove_cvref_t<Func>, Options>)
inline Summary Optimize(T &x, U &y, Rest &...rest, const Func &func,
                        const Options &options = {}) {
  ParamsPack pack(x, y, rest...);
  auto wrapped = detail::make_packed_adapter<Func, T, U, Rest...>(func);
  return Optimize(pack, wrapped, options);
}

}  // namespace tinyopt
