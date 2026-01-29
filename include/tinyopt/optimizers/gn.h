// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tinyopt/math.h>

#include <tinyopt/optimizers/optimizer.h>
#include <tinyopt/solvers/gn.h>

/// Gauss-Newton specific solver, optimizer and their options
namespace tinyopt::gn {

/// Gauss-Newton Solver
template <typename Hessian_t>
using Solver = solvers::SolverGN<Hessian_t>;

/// Gauss-Newton Sparse Solver
template <typename Hessian_t = SparseMat>
using SparseSolver = solvers::SolverGN<Hessian_t>;

/// Gauss-Newton Optimizer
template <typename Hessian_t>
using Optimizer = Optimizer_<solvers::SolverGN<Hessian_t>>;

}  // namespace tinyopt::gn
