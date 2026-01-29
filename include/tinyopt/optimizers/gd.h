// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tinyopt/math.h>

#include <tinyopt/optimizers/optimizer.h>
#include <tinyopt/solvers/gd.h>

/// Gradient Descent specific solver, optimizer and their options
namespace tinyopt::gd {

/// Gradient Descent Solver
template <typename Gradient_t>
using Solver = solvers::SolverGD<Gradient_t>;

/// Gradient Descent Sparse Solver
template <typename Hessian_t = SparseMat>
using SparseSolver = solvers::SolverGD<Hessian_t>;

/// Gradient Descent Optimizer
template <typename Gradient_t>
using Optimizer = Optimizer_<solvers::SolverGD<Gradient_t>>;


}  // namespace tinyopt::gd
