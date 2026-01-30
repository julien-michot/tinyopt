// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tinyopt/math.h>

#include <tinyopt/optimizers/optimizer.h>
#include <tinyopt/solvers/lm.h>
#include <type_traits>

/// Levenberg-Marquardt specific solver, optimizer and their options
namespace tinyopt::lm {

/// Levenberg-Marquardt Solver
template <typename Hessian_t>
using Solver = solvers::SolverLM<Hessian_t>;

/// Levenberg-Marquardt Sparse Solver
template <typename Hessian_t = SparseMat>
using SparseSolver = solvers::SolverLM<Hessian_t>;

/// Levenberg-Marquardt Optimizer
template <typename Hessian_t>
using Optimizer = Optimizer_<solvers::SolverLM<Hessian_t>>;

}  // namespace tinyopt::lm
