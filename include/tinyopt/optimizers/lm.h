// Copyright (C) 2025 Julien Michot. All Rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <tinyopt/math.h>
#include <tinyopt/optimize.h>

#include <tinyopt/optimizers/options.h>
#include <tinyopt/solvers/lm.h>
#include <type_traits>

/// Levenberg-Marquardt specific solver, optimizer and their options
namespace tinyopt::lm {

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
  constexpr int Dims = traits::params_trait<X_t>::Dims;
  // Detect Hessian Type, if it's dense or sparse
  constexpr bool isDense =
      std::is_invocable_v<Res_t, const X_t &> ||
      std::is_invocable_v<Res_t, const X_t &, Vector<Scalar, Dims> &, Matrix<Scalar, Dims, Dims> &>;
  using Hessian_t = std::conditional_t<isDense, Matrix<Scalar, Dims, Dims>, SparseMatrix<Scalar>>;
  return tinyopt::Optimize<Optimizer<Hessian_t>>(x, func, options);
}

}  // namespace tinyopt::lm
