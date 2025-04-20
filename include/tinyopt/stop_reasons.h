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

#include <ostream>
#include <sstream>

#include <tinyopt/traits.h>

namespace tinyopt {

/// @brief The reason why the optimization terminated
enum StopReason : int {
  /**
   * @name Failures (negative enums)
   * @{
   */

  kOutOfMemory = -4,        ///< Out of memory when allocating the system (Hessian(s)
  kSolverFailed = -3,       ///< Failed to solve the normal equations (H is not invertible)
  kSystemHasNaNOrInf = -2,  ///< Residuals or Jacobians have NaNs or Infinity
  kSkipped = -1,            ///< The system has no residuals or nothing to optimize or H is all 0s

  /** @} */
  /**
   * @name Success (positive enums or 0)
   * @{
   */

  kNone = 0,         ///< No stop, used by Step() or when no iterations done  (success)
  kMinError,         ///< Minimal error reached  (success)
  kMinRelError,      ///< Minimal relative error decrease reached (success)
  kMinDeltaNorm,     ///< Minimal delta norm reached (success)
  kMinGradNorm,      ///< Minimal gradient reached (success)
  kMaxIters,         ///< Maximum number of iterations reached (success)
  kMaxNoDecr,        ///< Failed to decrease error too many times (success)
  kMaxConsecNoDecr,  ///< Failed to decrease error consecutively too many times (success)
  kTimedOut,         ///< Total allocated time reached (success)
  kUserStopped       ///< User stopped the process (success)

  /** @} */
};

/// Stop reason description
template <typename Output, typename Options = std::nullptr_t>
std::string StopReasonDescription(const Output &out, const Options &options = {}) {
  std::ostringstream os;
  switch (out.stop_reason) {
    case StopReason::kNone:
      os << "🌱 Optimization not ran or used with Step() (success)";
      break;
    /**
     * @name Successes + convergence
     * @{
     */
    case StopReason::kMinError:
      os << "🌞 Reached minimum error (success)";
      if constexpr (!traits::is_nullptr_v<Options>)
        os << " ε:[" << (double)out.final_cost << " < " << options.min_error << "]";
      break;
    case StopReason::kMinRelError:
      os << "🌞 Reached minimum relative error (success)";
      if constexpr (!traits::is_nullptr_v<Options>)
        os << " ε:[" << out.final_rerr_dec << " < " << options.min_rerr_dec << "]";
      break;
    case StopReason::kMinDeltaNorm:
      os << "🌞 Reached minimal delta norm (success)";
      if constexpr (!traits::is_nullptr_v<Options>) {
        if (out.deltas2.empty())
          os << " |δX|:[" << out.deltas2.back() << " < " << std::sqrt(options.min_step_norm2)
             << "]";
        else
          os << " [|δX| < " << std::sqrt(options.min_step_norm2) << "]";
      }
      break;
    case StopReason::kMinGradNorm:
      os << "🌞 Reached minimal gradient (success)";
      if constexpr (!traits::is_nullptr_v<Options>)
        os << " [|∇| < " << std::sqrt(options.min_grad_norm2) << "]";
      break;
    /** @} */
    /**
     * @name Successes - no convergence
     * @{
     */
    case StopReason::kMaxIters:
      os << "⛅ Reached maximum number of iterations (success)";
      if constexpr (!traits::is_nullptr_v<Options>)
        os << " [#it == " << (int)options.max_iters << "]";
      break;
    case StopReason::kMaxNoDecr:
      os << "⛅ Failed to decrease error too many times (success)";
      if constexpr (!traits::is_nullptr_v<Options>)
        os << " [=" << (int)options.max_total_failures << "]";
      break;
    case StopReason::kMaxConsecNoDecr:
      os << "⛅ Failed to decrease error consecutively too many times (success)";
      if constexpr (!traits::is_nullptr_v<Options>)
        os << " [=" << (int)options.max_consec_failures << "]";
      break;
    case StopReason::kTimedOut:
      os << "⌛ Reached maximum allocated time (success)";
      if constexpr (!traits::is_nullptr_v<Options>)
        os << " τ:[" << out.duration_ms << " > " << options.max_duration_ms << "ms]";
      break;
    case StopReason::kUserStopped:
      os << "👍 User stopped the process (success)";
      break;
      /** @} */

      /**
       * @name Failures
       * @{
       */
    case StopReason::kOutOfMemory:
      os << "❌ Out of memory when allocating the Hessian(s), use SparseMatrix? (failure)";
      break;
    case StopReason::kSystemHasNaNOrInf:
      os << "❌ Residuals or Jacobians have NaNs or Inf (failure)";
      break;
    case StopReason::kSolverFailed:
      os << "❌ Failed to solve the normal equations (failure)";
      break;
    case StopReason::kSkipped:
      os << "❌ The system has no residuals or nothing to optimize (failure)";
      break;
      /** @} */
    default:
      os << "⛈️ Unknown reason:" << (int)out.stop_reason;
      break;
  }
  return os.str();
}

}  // namespace tinyopt
