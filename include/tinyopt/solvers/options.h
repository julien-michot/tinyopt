// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tinyopt::solvers {

/***
 *  @brief Common Solver Optimization options for 1st order methods
 *
 ***/
struct Options1 {
  Options1() {};

  /// Gradient clipping to range [-v, +v], disabled if 0
  float grad_clipping = 0;

  /**
   * @name Cost scaling options (mostly for NLLS solvers really)
   * @{
   */
   struct CostScaling {
    bool use_squared_norm = true;  ///< Use squared norm instead of norm (faster)
    bool downscale_by_2 = false;   ///< Rescale the cost by 0.5
    /// Normalize the final error by the number of residuals (after use_squared_norm)
    bool normalize = false;
  } cost;

  /** @} */

  struct {
    bool enable = true;  // Enable solver logging
    bool print_failure = false;  // Log when a failure to solve the linear system happens
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

  bool use_ldlt = true;   ///< If not, will use H.inverse() without any checks on invertibility
                          ///< except for Dims==1
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
