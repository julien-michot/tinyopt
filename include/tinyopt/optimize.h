// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tinyopt/optimizers/optimizer.h>
#include <tinyopt/optimizers/options.h>

#include <tinyopt/optimizers/optimizers.h>
#include <tuple>
#include "tinyopt/log.h"

namespace tinyopt {

namespace detail {

template <typename T>
using remove_cvref_t = std::remove_cv_t<std::remove_reference_t<T>>;

template <typename T>
inline constexpr bool is_options_v = std::is_same_v<remove_cvref_t<T>, Options>;

template <typename Tuple, std::size_t... Is>
auto select_tuple(Tuple &&tuple, std::index_sequence<Is...>) {
  return std::forward_as_tuple(std::get<Is>(std::forward<Tuple>(tuple))...);
}

template <typename Func, typename... Params>
inline Output optimize_from_params(Func &&func, const Options &options, Params &&...params) {
  auto flat = tinyopt::detail::flatten_parameters(params...);
  auto refs = std::forward_as_tuple(params...);
  auto wrapped = [func = std::forward<Func>(func), refs](const auto &flat_x) {
    auto current = refs;
    std::apply([&](auto &...args) { tinyopt::detail::restore_parameters(flat_x, args...); }, current);
    return std::apply(func, current);
  };
  auto out = Optimize(flat, wrapped, options);
  tinyopt::detail::restore_parameters(flat, params...);
  return out;
}

}  // namespace detail

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

template <typename T, typename U, typename... Rest, typename Func>
  requires(!std::is_same_v<std::remove_cvref_t<Func>, Options>)
inline Output Optimize(T &x, U &y, Rest &...rest, const Func &func, const Options &options = {}) {
  auto flat = tinyopt::detail::flatten_parameters(x, y, rest...);
  const auto wrapped = [&](const auto &flat_x) {
    using FlatItem = std::decay_t<decltype(flat_x[0])>;
    auto local = std::tuple<
        std::decay_t<decltype(tinyopt::traits::params_trait<std::remove_cvref_t<T>>::template cast<
            FlatItem>(std::declval<const std::remove_cvref_t<T> &>()))>,
        std::decay_t<decltype(tinyopt::traits::params_trait<std::remove_cvref_t<U>>::template cast<
            FlatItem>(std::declval<const std::remove_cvref_t<U> &>()))>,
        std::decay_t<decltype(tinyopt::traits::params_trait<std::remove_cvref_t<Rest>>::template cast<
            FlatItem>(std::declval<const std::remove_cvref_t<Rest> &>()))>...>{
        tinyopt::traits::params_trait<std::remove_cvref_t<T>>::template cast<FlatItem>(x),
        tinyopt::traits::params_trait<std::remove_cvref_t<U>>::template cast<FlatItem>(y),
        tinyopt::traits::params_trait<std::remove_cvref_t<Rest>>::template cast<FlatItem>(rest)...};
    std::apply([&](auto &...args) { tinyopt::detail::restore_parameters(flat_x, args...); }, local);
    return std::apply(func, local);
  };
  auto out = Optimize(flat, wrapped, options);
  tinyopt::detail::restore_parameters(flat, x, y, rest...);
  return out;
}

}  // namespace tinyopt
