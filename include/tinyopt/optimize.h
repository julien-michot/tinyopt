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
