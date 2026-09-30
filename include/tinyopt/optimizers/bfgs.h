// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tinyopt/optimizers/optimizer.h>
#include <tinyopt/solvers/bfgs.h>

namespace tinyopt::bfgs {

template <typename Gradient_t>
using Solver = solvers::SolverBFGS<Gradient_t>;

template <typename Gradient_t>
using Optimizer = Optimizer_<solvers::SolverBFGS<Gradient_t>>;

}  // namespace tinyopt::bfgs

namespace tinyopt::lbfgs {

template <typename Gradient_t, std::size_t HistoryCapacity = 8>
using Solver = solvers::SolverLBFGS<Gradient_t, HistoryCapacity>;

template <typename Gradient_t, std::size_t HistoryCapacity = 8>
using Optimizer = Optimizer_<solvers::SolverLBFGS<Gradient_t, HistoryCapacity>>;

}  // namespace tinyopt::lbfgs