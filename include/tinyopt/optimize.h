// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tinyopt/optimizers/optimizer.h>

namespace tinyopt {

/// Simplest interface to optimize `x` and minimize residuals (loss function).
/// Internally call the optimizer and run the optimization.
template <typename Optimizer, typename X_t, typename Res_t>
inline auto Optimize(X_t &x, const Res_t &func, const typename Optimizer::Options &options = {}) {
  Optimizer optimizer(options);
  return optimizer(x, func);
}

}  // namespace tinyopt
