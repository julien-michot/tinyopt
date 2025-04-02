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
#include <tinyopt/traits.h>

/// Default Optimize interface for generic unconstrained problems
namespace tinyopt {

template <typename SolverType, typename X_t, typename Res_t>
inline auto Optimize(X_t &x, const Res_t &func,
                     const typename optimizers::Optimizer<SolverType>::Options &options =
                         typename optimizers::Optimizer<SolverType>::Options()) {
  using Optimizer = optimizers::Optimizer<SolverType>;
  Optimizer optimizer(options);
  return optimizer(x, func);
}

}  // namespace tinyopt
