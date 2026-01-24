// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tinyopt/math.h>
#include <tinyopt/optimize.h>

#include <tinyopt/solvers/gd.h>

/// Gradient Descent specific solver, optimizer and their options
namespace tinyopt::gd {

/// Gradient Descent Optimization Options
struct Options : Options1 {
  Options(const Options1 options = {}) : Options1{options} {}
  gd::SolverOptions solver;
};

/// Gradient Descent Solver
template <typename Gradient_t>
using Solver = solvers::SolverGD<Gradient_t>;

/// Gradient Descent Optimizater type
template <typename Gradient_t>
using Optimizer = optimizers::Optimizer<Solver<Gradient_t>, Options>;

/// Gradient Descent Optimize function
template <typename X_t, typename Res_t>
inline auto Optimize(X_t &x, const Res_t &func, const Options &options = Options()) {
  using Scalar = std::conditional_t<
      std::is_scalar_v<typename traits::params_trait<X_t>::Scalar>,
      typename traits::params_trait<X_t>::Scalar,
      typename traits::params_trait<typename traits::params_trait<X_t>::Scalar>::Scalar>;
  static_assert(std::is_scalar_v<Scalar>);
  constexpr Index Dims = traits::params_trait<X_t>::Dims;
  // Detect Hessian Type, if it's dense or sparse
  constexpr bool isDense = std::is_invocable_v<Res_t, const X_t &> ||
                           std::is_invocable_v<Res_t, const X_t &, Vector<Scalar, Dims> &>;
  using Gradient_t = std::conditional_t<isDense, Vector<Scalar, Dims>, SparseMatrix<Scalar>>;

  static_assert(Solver<Gradient_t>::FirstOrder);
  return tinyopt::Optimize<Optimizer<Gradient_t>>(x, func, options);
}

}  // namespace tinyopt::gd
