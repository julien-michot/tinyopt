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

namespace tinyopt::solvers {

/***
 *  @brief Common Solver Optimization options for 1st order methods
 *
 ***/
struct Solver1Options {
  Solver1Options() {};

  struct {
    bool enable = true;  // Enable solver logging
  } log;
};

/***
 *  @brief Common Solver Optimization options for 2nd  or pseudo 2nd order methods
 *
 ***/
struct Solver2Options : Solver1Options {
  Solver2Options(const Solver1Options &options = {}) : Solver1Options{options} {}

  /**
   * @name H Properties
   * @{
   */

  bool ldlt = true;       ///< If not, will use H.inverse()
  bool H_is_full = true;  ///< Specify if H is only Upper triangularly or fully filled

  /** @} */
};
}  // namespace tinyopt::solvers
