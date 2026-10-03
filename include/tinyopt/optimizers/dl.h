// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tinyopt/optimizers/optimizer.h>
#include <tinyopt/solvers/dl.h>

namespace tinyopt::dl {

template <typename Hessian_t>
using Solver = solvers::SolverDogLeg<Hessian_t>;

template <typename Hessian_t>
using Optimizer = Optimizer_<solvers::SolverDogLeg<Hessian_t>>;

}  // namespace tinyopt::dl