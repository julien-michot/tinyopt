// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tinyopt/optimizers/nlls.h>

namespace tinyopt::benchmark {

inline auto CreateOptions(Options::Solver solver = Options::Solver::LevenbergMarquardt,
                          bool enable_log = false) {
  Options options(solver);
  options.stop.max_iters = 100;

  options.stop.min_error = 1e-12f;
  options.stop.min_rerr_dec = 1e-6f;
  options.stop.min_step_norm2 = 1e-16f;
  options.stop.min_grad_norm2 = 0.0f;
  options.stop.min_grad_max = 1e-9f;

  // Allow LM sufficient consecutive damping increases on difficult steps.
  options.stop.max_consec_failures = 10;

  // No log?
  options.log.enable = enable_log;
  options.exp.final_cov = false;
  options.exp.save_history = false;
  options.exp.measure_time = false;
  return options;
}

}  // namespace tinyopt::benchmark