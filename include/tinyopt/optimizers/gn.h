// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tinyopt/math.h>
#include <tinyopt/optimize.h>

#include <tinyopt/solvers/gn.h>

/// Gauss-Newton specific solver, optimizer and their options
namespace tinyopt::nlls::gn {

/// Gauss-Newton Optimization Options
struct Options : Options2 {
  Options(const Options2 options = {}) : Options2{options} {}
  gn::SolverOptions solver;
};

/// Gauss-Newton Solver
template <typename Hessian_t>
using Solver = solvers::SolverGN<Hessian_t>;

/// Gauss-Newton Sparse Solver
template <typename Hessian_t = SparseMatrix<double>>
using SparseSolver = solvers::SolverGN<Hessian_t>;

/// Gauss-Newton Optimizater type
template <typename Hessian_t>
using Optimizer = optimizers::Optimizer<Solver<Hessian_t>, Options>;

/// Gauss-Newton Optimize function
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

}  // namespace tinyopt::nlls::gn
