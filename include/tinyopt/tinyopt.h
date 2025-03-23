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

#include "tinyopt/lm.h" // so far, only LM

namespace tinyopt {

// By default, Optimize, Options and Output are from LM method

using Options = lm::Options;

template <typename JtJ_t> using Output = lm::Output<JtJ_t>;

/***
 *  @brief Minimize a loss function @arg acc using the Levenberg-Marquardt
 *  minimization algorithm.
 *
 ***/
template <typename ParametersType, typename ResidualsFunc>
inline auto Optimize(ParametersType &x, ResidualsFunc &func,
                     const Options &options = Options{}) {
  return lm::LM(x, func, options);
}

} // namespace tinyopt