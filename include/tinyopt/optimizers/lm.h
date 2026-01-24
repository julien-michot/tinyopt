// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tinyopt/math.h>
#include <tinyopt/optimize.h>

#include <tinyopt/optimizers/options.h>
#include <tinyopt/solvers/lm.h>
#include <type_traits>

/// Levenberg-Marquardt specific solver, optimizer and their options
namespace tinyopt::nlls::lm {

/// Levenberg-Marquardt Optimization Options
struct Options : Options2 {
  Options(const Options2 options = {}) : Options2{options} {}
  lm::SolverOptions solver;
};

/// Levenberg-Marquardt Solver
template <typename Hessian_t>
using Solver = solvers::SolverLM<Hessian_t>;

/// Levenberg-Marquardt Sparse Solver
template <typename Hessian_t = SparseMatrix<double>>
using SparseSolver = solvers::SolverLM<Hessian_t>;

/// Levenberg-Marquardt Optimizater type
template <typename Hessian_t>
using Optimizer = optimizers::Optimizer<Solver<Hessian_t>, Options>;

/// Levenberg-Marquardt Optimize function
template <typename X_t, typename Res_t>
inline auto Optimize(X_t &x, const Res_t &func, const Options &options = Options()) {
  // Detect Scalar, supporting at most one nesting level
  using Scalar = std::conditional_t<
      std::is_scalar_v<typename traits::params_trait<X_t>::Scalar>,
      typename traits::params_trait<X_t>::Scalar,
      typename traits::params_trait<typename traits::params_trait<X_t>::Scalar>::Scalar>;
  static_assert(std::is_scalar_v<Scalar>);
  constexpr Index Dims = traits::params_trait<X_t>::Dims;
  // Detect Hessian Type, if it's dense or sparse
  constexpr bool isDense =
      std::is_invocable_v<Res_t, const X_t &> ||
      std::is_invocable_v<Res_t, const X_t &, Vector<Scalar, Dims> &, Matrix<Scalar, Dims, Dims> &>;
  using Hessian_t = std::conditional_t<isDense, Matrix<Scalar, Dims, Dims>, SparseMatrix<Scalar>>;
  return tinyopt::Optimize<Optimizer<Hessian_t>>(x, func, options);
}

}  // namespace tinyopt::nlls::lm
