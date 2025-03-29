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

#include <cassert>

#include <tinyopt/log.h>
#include <tinyopt/math.h>
#include <tinyopt/opt_jet.h>
#include <tinyopt/options.h>
#include <tinyopt/output.h>
#include <tinyopt/traits.h>

namespace tinyopt::lm {

/***
 *  @brief LM Optimization options
 *
 ***/
struct Options : tinyopt::CommonOptions {
  Options(const tinyopt::CommonOptions &options = {}) : tinyopt::CommonOptions(options) {}
  double damping_init = 1e-4;  ///< Initial damping factor. If 0, the damping is disable (it will
                               ///< behave like Gauss-Newton)
  ///< Min and max damping values (only used when damping_init != 0)
  std::array<double, 2> damping_range{{1e-9, 1e9}};
};

/***
 *  @brief LM Optimization Output (same as Gauss-Newton's)
 *
 ***/
template <typename JtJ_t>
using Output = tinyopt::Output<JtJ_t>;

/***
 *  @brief Minimize a loss function @arg acc using the Levenberg-Marquardt minimization algorithm.
 *
 ***/
template <typename ParametersType, typename ResidualsFunc>
inline auto LM(ParametersType &X, const ResidualsFunc &acc, const Options &options = Options{}) {
  using std::sqrt;
  using ptrait = traits::params_trait<ParametersType>;

  using Scalar = typename ptrait::Scalar;
  constexpr int Size = ptrait::Dims;

  int size = Size;  // Dynamic size
  if constexpr (Size == Dynamic) size = ptrait::dims(X);

  using JtJ_t = Matrix<Scalar, Size, Size>;
  using OutputType = Output<JtJ_t>;
  OutputType out;
  if (size == Dynamic || size == 0) {
    TINYOPT_LOG("Error: Parameters dimensions cannot be 0 or Dynamic");
    out.stop_reason = StopReason::kSkipped;
    return out;
  }

  const std::string e_str = options.log.print_rmse ? "√ε²/n" : "ε²";

  bool already_rolled_true = true;
  const uint8_t max_tries =
      options.max_consec_failures > 0 ? std::max<uint8_t>(1, options.max_total_failures) : 255;
  auto X_last_good = X;
  double lambda = options.damping_init;
  out.errs2.reserve(out.num_iters + 2);
  out.deltas2.reserve(out.num_iters + 2);
  out.successes.reserve(out.num_iters + 2);
  if (options.export_JtJ) out.last_JtJ = JtJ_t::Zero(size, size);
  JtJ_t JtJ(size, size);
  Matrix<Scalar, Size, 1> Jt_res(size, 1);
  Matrix<Scalar, Size, 1> dX(size, 1);
  for (; out.num_iters < options.num_iters + 1 /*+1 to potentially roll-back*/; ++out.num_iters) {
    JtJ.setZero();
    Jt_res.setZero();

    // Update JtJ and Jt_res by accumulating changes
    const auto &output = acc(X, JtJ, Jt_res);

    double err;    // accumulated error (for monotony check and logging)
    int nerr = 1;  // number of residuals (optional, for logging)

    using ResOutputType = std::remove_const_t<std::remove_reference_t<decltype(output)>>;
    if constexpr (traits::is_pair_v<ResOutputType>) {
      using ResOutputType1 =
          std::remove_const_t<std::remove_reference_t<decltype(std::get<0>(output))>>;
      if constexpr (traits::is_matrix_or_array_v<ResOutputType1>) {
        err = std::get<0>(output).squaredNorm();
        if (std::get<0>(output).size() == 0) nerr = 0;
      } else
        err = std::get<0>(output);
      nerr = std::get<1>(output);
    } else if constexpr (std::is_scalar_v<ResOutputType>) {
      err = output;
    } else if constexpr (traits::is_matrix_or_array_v<ResOutputType>) {
      err = output.squaredNorm();
      if (output.size() == 0) nerr = 0;
    } else {
      // You're not returning a supported type (must be float, double or Matrix)
      static_assert(traits::is_matrix_or_array_v<ResOutputType> || std::is_scalar_v<ResOutputType>);
    }

    const bool skip_solver = nerr == 0;
    out.num_residuals = nerr;
    if (nerr == 0) {
      if (options.log.enable) TINYOPT_LOG("❌ #{}: No residuals, stopping", out.num_iters);
      // Can break only if first time, otherwise better count it as failure
      if (out.num_iters == 0) {
        out.stop_reason = StopReason::kSkipped;
        break;
      }
    } else if (std::isnan(err) || std::isinf(err)) {
      if (options.log.enable) TINYOPT_LOG("❌ #{}: NaN/Inf in error", out.num_iters);
      // Can break only if first time, otherwise better count it as failure
      if (out.num_iters == 0) {
        out.stop_reason = StopReason::kSystemHasNaNOrInf;
        break;
      }
    }

    // Damping
    if (lambda > 0.0) {
      for (int i = 0; i < size; ++i) JtJ(i, i) *= (1.0 + lambda);
    }

    dX.setZero();
    bool solver_failed = skip_solver;
    for (; !skip_solver && out.num_consec_failures <= max_tries;) {
      // Solver linear system
      if (options.ldlt) {
        const auto dx_ = SolveAXb(JtJ, Jt_res);
        if (dx_) {
          dX = -dx_.value();
          solver_failed = false;
          break;
        } else {
          solver_failed = true;
          dX.setZero();
        }
      } else {  // Use default inverse
        // Fill the lower part of JtJ then inverse it
        if (!options.JtJ_is_full)
          JtJ.template triangularView<Lower>() = JtJ.template triangularView<Upper>().transpose();
        dX = -JtJ.inverse() * Jt_res;
        solver_failed = false;
        break;
      }
      // Sover failed -> re-do the damping
      if (options.damping_init > 0.0) {
        const double l =
            std::min(options.damping_range[1], std::max(options.damping_range[0], lambda * 10));
        if (options.log.enable)
          TINYOPT_LOG("❌ #{}: Cholesky Failed, redamping to λ:{:.2e}", out.num_iters, l);
        const double s = (1.0 + l) / (1.0 + lambda);  // rescaling factor
        for (int i = 0; i < size; ++i) JtJ(i, i) *= s;
        lambda = l;
      } else {  // Gauss-Newton -> no damping
        break;
      }
      out.num_consec_failures++;
      out.num_failures++;
    }

    // Check the displacement magnitude
    const double dX_norm2 = solver_failed ? 0 : dX.squaredNorm();
    const double Jt_res_norm2 = options.min_grad_norm2 == 0.0f ? 0 : Jt_res.squaredNorm();
    if (std::isnan(dX_norm2) || std::isinf(dX_norm2)) {
      solver_failed = true;
      if (options.log.print_failure) {
        TINYOPT_LOG("❌ Failure, dX = \n{}", dX.template cast<float>());
        TINYOPT_LOG("JtJ = \n{}", JtJ);
        TINYOPT_LOG("Jt*res = \n{}", Jt_res);
      }
      out.stop_reason = StopReason::kSystemHasNaNOrInf;
      break;
    }

    const double derr = err - out.last_err2;
    // Save history of errors and deltas
    out.errs2.emplace_back(err);
    out.deltas2.emplace_back(dX_norm2);
    // Convert X to string (if log enabled)
    std::string x_str;
    if (options.log.print_x) {
      std::ostringstream oss_x;
      if constexpr (traits::is_matrix_or_array_v<ParametersType>) {  // Flattened X
        oss_x << "X:[";
        if (X.cols() == 1)
          oss_x << X.transpose();
        else
          oss_x << X.reshaped().transpose();
        oss_x << "] ";
      } else if constexpr (traits::is_streamable_v<ParametersType>) {
        oss_x << "X:{" << X << "} ";  // User must define the stream operator of ParameterType
      }
      x_str = oss_x.str();
    }
    // Check step quality
    if (derr < 0.0 && !solver_failed) { /* GOOD Step */
      out.successes.emplace_back(true);
      if (out.num_iters > 0) X_last_good = X;
      // Move X by dX
      ptrait::pluseq(X, dX);
      // Save results
      out.last_err2 = err;
      if (options.export_JtJ) {
        out.last_JtJ = JtJ;  // TODO actually better store the one right after a success
        if (lambda > 0.0) {
          for (int i = 0; i < Size; ++i) out.last_JtJ(i, i) = JtJ(i, i) / (1.0f + lambda);
        }
      }
      already_rolled_true = false;
      out.num_consec_failures = 0;
      // Log
      if (options.log.enable) {
        const double e = options.log.print_rmse ? std::sqrt(err / nerr) : err;
        const double sigma = sqrt(InvCov(JtJ).value().maxCoeff());  // JtJ is invertible here!
        TINYOPT_LOG(
            "✅ #{}: {}|δX|:{:.2e} λ:{:.2e} ⎡σ⎤:{:.4f} {}:{:.5f} n:{} dε²:{:.3e} ∇ε²:{:.3e}",
            out.num_iters, x_str, sqrt(dX_norm2), lambda, sigma, e_str, e, nerr, derr,
            Jt_res_norm2);
      }

      if (options.damping_init > 0.0)
        lambda =
            std::min(options.damping_range[1], std::max(options.damping_range[0], lambda / 3.0));
    } else { /* BAD Step */
      out.successes.emplace_back(false);
      // Log
      if (options.log.enable) {
        const double e = options.log.print_rmse ? std::sqrt(err / nerr) : err;
        TINYOPT_LOG("❌ #{}: X:[{}] |δX|:{:.2e} λ:{:.2e} {}:{:.5f} n:{} dε²:{:.3e} ∇ε²:{:.3e}",
                    out.num_iters, x_str, sqrt(dX_norm2), lambda, e_str, e, nerr, derr,
                    Jt_res_norm2);
      }
      if (!already_rolled_true) {
        X = X_last_good;  // roll back by copy
        already_rolled_true = true;
      }
      out.num_failures++;
      out.num_consec_failures++;
      if (options.max_consec_failures > 0 &&
          out.num_consec_failures >= options.max_consec_failures) {
        out.stop_reason = StopReason::kMaxConsecFails;
        break;
      }
      if (options.max_total_failures > 0 && out.num_failures >= options.max_total_failures) {
        out.stop_reason = StopReason::kMaxFails;
        break;
      }
      if (options.damping_init > 0.0)
        lambda = std::min(options.damping_range[1], std::max(options.damping_range[0], lambda * 2));
      // TODO don't rebuild if no rollback!
    }

    // Detect if we need to stop and the reason
    if (solver_failed) {
      out.stop_reason = StopReason::kSolverFailed;
      break;
    } else if (options.min_delta_norm2 > 0 && dX_norm2 < options.min_delta_norm2) {
      out.stop_reason = StopReason::kMinDeltaNorm;
      break;
    } else if (options.min_grad_norm2 > 0 && Jt_res_norm2 < options.min_grad_norm2) {
      out.stop_reason = StopReason::kMinGradNorm;
      break;
    }
  }
  return out;
}

/**
 * @brief Minimize a loss function using the Levenberg-Marquardt algorithm.
 *
 * This function optimizes a set of parameters `x` to minimize a given loss function,
 * employing the Levenberg-Marquardt minimization algorithm.
 *
 * @tparam ParametersType Type of the parameters to be optimized. Must support arithmetic operations
 * and assignment.
 * @tparam ResidualsFunc Type of the residuals function. Must be callable with ParametersType and
 * return a scalar or a vector of residuals. The function signature is either f(x) or f(x, JtJ,
 * Jt_res).
 *
 * @param[in,out] x The initial and optimized parameters. Modified in-place.
 * @param[in] func The residual function to be minimized. It should return a vector of residuals
 * based on the input parameters.
 * @param[in] options Optional parameters for the optimization process (e.g., tolerances, max
 * iterations). Defaults to `Options{}`.
 *
 * @return The optimization details (`Output` struct).
 *
 * @code
 * // Example usage:
 * float x = 1;
 * const auto &out = Optimize(x, [](const auto &x) { return x * x - 2.0; });
 * @endcode
 */
template <typename ParametersType, typename ResidualsFunc>
inline auto Optimize(ParametersType &x, const ResidualsFunc &func,
                     const Options &options = Options{}) {
  if constexpr (std::is_invocable_v<ResidualsFunc, const ParametersType &>) {
    const auto optimize = [](auto &x, const auto &func, const auto &options) {
      return LM(x, func, options);
    };
    return tinyopt::OptimizeJet(x, func, optimize, options);
  } else {
    return LM(x, func, options);
  }
}

}  // namespace tinyopt::lm
