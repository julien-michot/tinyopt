// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tinyopt/optimizers/nlls.h>

namespace tinyopt::benchmark {

inline auto CreateOptions(bool enable_log = false) {
  Options options;
  options.stop.max_iters = 10;

  options.stop.min_error = 0;  // Ceres does not seem to be using this
  options.stop.min_rerr_dec = 1e-12f;
  options.stop.min_step_norm2 = 1e-16f;

  // Stops early if it's failing consecutively. Note: Tinyopt can have an extra failure at start
  // compared to Ceres.
  options.stop.max_consec_failures = 3;

  // No log?
  options.log.enable = enable_log;
  options.log.enable = enable_log;
  options.hessian.save_last = false;
  return options;
}

}  // namespace tinyopt::benchmark