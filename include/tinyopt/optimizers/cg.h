// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tinyopt/optimizers/optimizer.h>
#include <tinyopt/solvers/cg.h>

namespace tinyopt::cg {

template <typename Gradient_t>
using Solver = solvers::SolverCG<Gradient_t>;

template <typename Gradient_t>
using Optimizer = Optimizer_<solvers::SolverCG<Gradient_t>>;

}  // namespace tinyopt::cg