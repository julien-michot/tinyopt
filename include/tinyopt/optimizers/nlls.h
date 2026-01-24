// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tinyopt/optimizers/gd.h>

#include <tinyopt/optimizers/lm.h>

namespace tinyopt::nlls {

/// Default Optimizer and options for Non-linear Least Squares (`NLLS`) Optimization
/// Here we default to Levenberg-Marquardt algorithm.
using namespace tinyopt::nlls::lm;

}  // namespace tinyopt::nlls
