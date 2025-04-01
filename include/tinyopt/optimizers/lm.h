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

#include <tinyopt/solvers/lm.h>
#include "tinyopt/options.h"

/// Define convenient aliases for LM Optimizer
namespace tinyopt::lm {

/***
 *  @brief LM Optimization options
 *
 ***/
struct Options : CommonOptions2 {
  Options(const CommonOptions2 options = {}) : CommonOptions2{options} {}
  solvers::lm::SolverOptions solver;
};

template <typename Hessian_t = MatX>
using Solver = solvers::SolverLM<Hessian_t>;

template <typename Hessian_t = SparseMatrix<double>>
using SparseSolver = solvers::SolverLM<Hessian_t>;

template <typename Hessian_t = MatX>
using Optimizer = Optimizer<solvers::SolverLM<Hessian_t>>;

template <typename X_t, typename Res_t, int Dims = traits::params_trait<X_t>::Dims,
          typename SolverType =
              solvers::SolverLM<Matrix<typename traits::params_trait<X_t>::Scalar, Dims, Dims>>>
inline auto Optimize(X_t &x, const Res_t &func, const lm::Options &options = lm::Options()) {
  return Optimize<SolverType>(x, func, options);
}

}  // namespace tinyopt::lm
