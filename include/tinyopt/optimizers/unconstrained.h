// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tinyopt/optimizers/gd.h>

namespace tinyopt::guc {

/// Default Optimizer and options for general unconstrained (`guc`) optimization
/// Here we default to Gradient-Descent algorithm.
using namespace tinyopt::gd;

}  // namespace tinyopt::guc
