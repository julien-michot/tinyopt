// Copyright (C) 2025 Julien Michot. All Rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <tinyopt/optimizers/nlls.h>

namespace tinyopt::benchmark {

inline auto CreateOptions(bool enable_log = false) {
  tinyopt::nlls::lm::Options options;
  options.max_iters = 10;

  options.min_error = 0;  // Ceres does not seem to be using this
  options.min_rerr_dec = 1e-12f;
  options.min_step_norm2 = 1e-16f;

  // Stops early if it's failing consecutively. Note: Tinyopt can have an extra failure at start
  // compared to Ceres.
  options.max_consec_failures = 3;

  // No log?
  options.log.enable = enable_log;
  options.solver.log.enable = enable_log;
  options.save.H = false;
  return options;
}

}  // namespace tinyopt::benchmark