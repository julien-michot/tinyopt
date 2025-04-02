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

#include <tinyopt/solvers/gd.h>

/// Gradient Descent specific solver, optimizer and their options
namespace tinyopt::gd {

/// Gradient Descent Optimization Options
struct Options : CommonOptions {
  Options(const CommonOptions options = {}) : CommonOptions{options} {}
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
  return tinyopt::Optimize<Optimizer>(x, func, options);
}

}  // namespace tinyopt::gd
