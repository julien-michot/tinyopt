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

#include <cstdint>
#include <iostream>
#include <optional>
#include <ostream>
#include <sstream>
#include <vector>

#include <Eigen/Core>

#include <tinyopt/math.h>    // Defines Matrix and Vector
#include <tinyopt/traits.h>  // Defines parameters_size_v

namespace tinyopt {

/***
 *  @brief Struct containing optimization results
 *
 ***/
template <typename JtJ_t>
struct Output {
  enum StopReason : uint8_t {
    kMaxIters = 0,    ///< Reached maximum number of iterations (success)
    kMinDeltaNorm,    ///< Reached minimal delta norm (success)
    kMinGradNorm,     ///< Reached minimal gradient (success)
    kMaxFails,        ///< Failed to decrease error too many times (success)
    kMaxConsecFails,  ///< Failed to decrease error consecutively too many times (success)
    /// Failures
    kSystemHasNaNs,  ///< Residuals or Jacobians have NaNs
    kSolverFailed,   ///< Failed to solve the normal equations (inverse JtJ)
    kNoResiduals     ///< The system has no residuals
  };

  /// Last valid step results
  float last_err2 = std::numeric_limits<float>::max();

  /// Stop reason
  StopReason stop_reason = StopReason::kMaxIters;

  /// Stop reason description
  std::string StopReasonDescription() const {
    std::ostringstream os;
    switch (stop_reason) {
      // Succeess
      case StopReason::kMaxIters:
        os << "Reached maximum number of iterations (success)";
        break;
      case StopReason::kMinDeltaNorm:
        os << "Reached minimal delta norm (success)";
        break;
      case StopReason::kMinGradNorm:
        os << "Reached minimal gradient (success)";
        break;
      case StopReason::kMaxFails:
        os << "Failed to decrease error too many times (success)";
        break;
      case StopReason::kMaxConsecFails:
        os << "Failed to decrease error consecutively too many times (success)";
        break;
      // Failures
      case StopReason::kSystemHasNaNs:
        os << "Residuals or Jacobians have NaNs (failure)";
        break;
      case StopReason::kSolverFailed:
        os << "Failed to solve the normal equations (failure)";
        break;
      case StopReason::kNoResiduals:
        os << "The system has no residuals (failure)";
        break;
      default:
        os << "Unknown reason:" << (int)stop_reason;
        break;
    }
    return os.str();
  }

  /// Returns true if the stop reason is not a failure to solve or NaNs or missing residuals
  bool Succeeded() const {
    return stop_reason != StopReason::kSystemHasNaNs && stop_reason != StopReason::kSolverFailed &&
           stop_reason != StopReason::kNoResiduals;
  }

  /// Returns true if the optimization reached the specified minimal delta norm or gradient norm
  bool Converged() const {
    return stop_reason == StopReason::kMinDeltaNorm || stop_reason == StopReason::kMinGradNorm;
  }

  /// Returns an approximation of the covariance, namely JtJ.inverse(), if JtJ is invertible.
  /// Optionnaly rescale the covariance by ε²*(#ε - dims) (to be used when e.g. observations did not have noise modelling)
  std::optional<JtJ_t> Covariance(bool rescaled = false) const {
    JtJ_t cov = InvCov(last_JtJ);
    if (rescaled && num_residuals > last_JtJ.cols()) {
      return cov * (last_err2 / (num_residuals - last_JtJ.cols()));
    } else {
      return 0.5 * cov; // since Hessian approx is H = 2*JtJ^-1
    }
  }

  uint16_t num_residuals = 0;  ///< Final number of residuals
  uint16_t num_iters = 0;      ///< Final number of iterations
  uint8_t num_failures = 0;    ///< Final number of failures to decrease the error
  uint8_t num_consec_failures =
      0;           ///< Final number of the last consecutive failures to decrease the error
  JtJ_t last_JtJ;  ///< Final JtJ, including damping

  /// Per iteration results
  std::vector<float> errs2;     ///< Mean squared accumulated errors of all iterations
  std::vector<float> deltas2;   ///< Step sizes of all iterations
  std::vector<bool> successes;  ///< Step acceptation status for all iterations
};

}  // namespace tinyopt
