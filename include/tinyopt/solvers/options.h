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
struct Options1 {
  Options1() {};

  struct {
    bool enable = true;  // Enable solver logging
  } log;
};

/***
 *  @brief Common Solver Optimization options for 2nd  or pseudo 2nd order methods
 *
 ***/
struct Options2 : Options1 {
  Options2(const Options1 &options = {}) : Options1{options} {}

  /**
   * @name H Properties
   * @{
   */

  bool use_ldlt = true;   ///< If not, will use H.inverse()
  bool H_is_full = true;  ///< Specify if H is only Upper triangularly or fully filled

  /** @} */

  /**
   * @name Checks
   * @{
   */

  float check_min_H_diag = 0;  ///< Check the the hessian's diagonal are not all below the
                               ///< threshold. Use 0 to disable the check.

  /** @} */
};
}  // namespace tinyopt::solvers
