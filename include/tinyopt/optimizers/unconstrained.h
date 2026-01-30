// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tinyopt/optimizers/gd.h>

namespace tinyopt::unconstrained {

/// Default Optimizer and options for general unconstrained optimization
/// Here we default to Gradient-Descent algorithm.
using namespace tinyopt::gd;

}  // namespace tinyopt::unconstrained