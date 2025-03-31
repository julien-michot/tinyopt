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
#include <sstream>
#include <type_traits>

#include <tinyopt/log.h>
#include <tinyopt/math.h>
#include <tinyopt/opt_jet.h>
#include <tinyopt/options.h>
#include <tinyopt/output.h>
#include <tinyopt/time.h>
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

/**
 * @brief Minimize a loss function using the Levenberg-Marquardt algorithm.
 *
 * This function optimizes a set of parameters `x` to minimize a given loss function,
 * employing the Levenberg-Marquardt minimization algorithm.
 *
 * @tparam X_t Type of the parameters to be optimized. Must support arithmetic operations
 * and assignment.
 * @tparam Res_t Type of the residuals function. Must be callable with X_t and
 * return a scalar or a vector of residuals. The function signature must be f(x, grad, H)
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
 * const auto &out = LM(x, [](const auto &x, auto &grad, auto &H) {
 *    float res = x * x - 2.0;
 *    grad(0) = 2 * x * res;
 *    H(0) = 2 * x * 2 * x;
 *    return res;
 * });
 * @endcode
 */
template <typename X_t, typename AccFunc>
inline auto LM(X_t &x, const AccFunc &acc, const Options &options = Options{}) {
  using std::sqrt;
  using ptrait = traits::params_trait<X_t>;

  using Scalar = std::conditional_t<
      std::is_scalar_v<typename ptrait::Scalar>, typename ptrait::Scalar,  // Scalar
      typename traits::params_trait<typename ptrait::Scalar>::Scalar>;  // nested, only support one
                                                                        // level
  constexpr int Size = ptrait::Dims;
  int size = Size;  // Dynamic size
  if constexpr (Size == Dynamic) size = ptrait::dims(x);

  const auto t = tic();

  // Recover the Hessian type (Dense of Sparse)
  using H_t = std::conditional_t<std::is_invocable_v<AccFunc, const X_t &, Vector<Scalar, Size> &,
                                                     Matrix<Scalar, Size, Size> &>,
                                 Matrix<Scalar, Size, Size>, SparseMatrix<Scalar>>;
  // Define the output type
  using OutputType = Output<H_t>;
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
  auto X_last_good = x;
  double lambda = options.damping_init;
  out.errs2.reserve(out.num_iters + 2);
  out.deltas2.reserve(out.num_iters + 2);
  out.successes.reserve(out.num_iters + 2);

  // Create the Hessian
  H_t H; // TODO we can save memory if we use OutputType::last_H directly. Consider refactoring.

  // Check whether we can allocate H if it's dynamic sized
  if constexpr (Size == Dynamic || traits::is_sparse_matrix_v<H_t>) {
    try {
      H.resize(size, size);
      if (options.export_H) out.last_H.resize(size, size);
    } catch (const std::bad_alloc &e) {
      if (options.log.enable) {
        TINYOPT_LOG(
            "Failed to allocate {} Hessian(s) of size {}x{}, "
            "mem:{}GB, maybe use a SparseMatrix?",
            options.export_H ? 2 : 1, size, size, 1e-9 * size * size * sizeof(Scalar));
      }
      out.stop_reason = StopReason::kOutOfMemory;
      return out;
    }
  }
  if (options.export_H) out.last_H.setZero();

  // Create the gradient and displacement `dx`
  Matrix<Scalar, Size, 1> grad(size, 1);
  Matrix<Scalar, Size, 1> dx(size, 1);
  for (; out.num_iters < options.num_iters + 1 /*+1 to potentially roll-back*/; ++out.num_iters) {
    H.setZero();
    grad.setZero();

    // Update H and grad by accumulating changes
    const auto &output = acc(x, grad, H);

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
      for (int i = 0; i < size; ++i) {
        if constexpr (traits::is_matrix_or_array_v<H_t>)
          H(i, i) *= 1.0 + lambda;
        else {
          H.coeffRef(i, i) *= 1.0 + lambda;
        }
      }
    }

    dx.setZero();
    bool solver_failed = skip_solver;
    for (; !skip_solver && out.num_consec_failures <= max_tries;) {
      // Check whether it's taking too much time
      if (options.max_duration_ms > 0 && toc_ms(t) > options.max_duration_ms) {
        break;
      }
      // Solver linear system
      if (options.ldlt || traits::is_sparse_matrix_v<H_t>) {
        const auto dx_ = Solve(H, grad);
        if (dx_) {
          dx = -dx_.value();
          solver_failed = false;
          break;
        } else {
          solver_failed = true;
          dx.setZero();
        }
      } else if constexpr (!traits::is_sparse_matrix_v<H_t>) {  // Use default inverse
        // Fill the lower part of H then inverse it
        if (!options.H_is_full)
          H.template triangularView<Lower>() = H.template triangularView<Upper>().transpose();
        dx = -H.inverse() * grad;
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
        for (int i = 0; i < size; ++i) {
          if constexpr (traits::is_matrix_or_array_v<H_t>)
            H(i, i) *= s;
          else {
            H.coeffRef(i, i) *= s;
          }
        }

        lambda = l;
      } else {  // Gauss-Newton -> no damping
        break;
      }
      out.num_consec_failures++;
      out.num_failures++;
    }

    // Check the displacement magnitude
    const double dX_norm2 = solver_failed ? 0 : dx.squaredNorm();
    const double grad_norm2 = options.min_grad_norm2 == 0.0f ? 0 : grad.squaredNorm();
    if (std::isnan(dX_norm2) || std::isinf(dX_norm2)) {
      solver_failed = true;
      if (options.log.print_failure) {
        TINYOPT_LOG("❌ Failure, dX = \n{}", dx.template cast<float>());
        TINYOPT_LOG("H = \n{}", H);
        TINYOPT_LOG("grad = \n{}", grad);
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
      if constexpr (traits::is_matrix_or_array_v<X_t>) {  // Flattened X
        oss_x << " X:[";
        if (x.cols() == 1)
          oss_x << x.transpose();
        else
          oss_x << x.reshaped().transpose();
        oss_x << "]";
      } else if constexpr (traits::is_streamable_v<X_t>) {
        oss_x << " X:{" << x << "}";  // User must define the stream operator of ParameterType
      }
      x_str = oss_x.str();
    }
    // Check step quality
    if (derr < 0.0 && !solver_failed) { /* GOOD Step */
      out.successes.emplace_back(true);
      if (out.num_iters > 0) X_last_good = x;
      // Move X by dX
      ptrait::pluseq(x, dx);
      // Save results
      out.last_err2 = err;
      if (options.export_H) {
        out.last_H = H;  // TODO actually better store the one right after a success
        if (lambda > 0.0) {
          for (int i = 0; i < size; ++i) {
            if constexpr (traits::is_matrix_or_array_v<H_t>)
              out.last_H(i, i) = H(i, i) / (1.0f + lambda);
            else
              out.last_H.coeffRef(i, i) = H.coeffRef(i, i) / (1.0f + lambda);
          }
        }
      }
      already_rolled_true = false;
      out.num_consec_failures = 0;
      // Log
      if (options.log.enable) {
        const double e = options.log.print_rmse ? std::sqrt(err / nerr) : err;
        std::ostringstream oss_sigma;
        if (options.log.print_max_stdev) {
          if constexpr (traits::is_sparse_matrix_v<H_t>)
            oss_sigma << " ⎡σ⎤:" << sqrt(InvCov(H).value().coeffs().maxCoeff());
          else
            oss_sigma << " ⎡σ⎤:" << sqrt(InvCov(H).value().maxCoeff());
        }
        TINYOPT_LOG("✅ #{}:{} |δX|:{:.2e} λ:{:.2e}{} {}:{:.5f} n:{} dε²:{:.3e} ∇ε²:{:.3e}",
                    out.num_iters, x_str, sqrt(dX_norm2), lambda, oss_sigma.str(), e_str, e, nerr,
                    derr, grad_norm2);
      }

      if (options.damping_init > 0.0)
        lambda =
            std::min(options.damping_range[1], std::max(options.damping_range[0], lambda / 3.0));
    } else { /* BAD Step */
      out.successes.emplace_back(false);
      // Log
      if (options.log.enable) {
        const double e = options.log.print_rmse ? std::sqrt(err / nerr) : err;
        TINYOPT_LOG("❌ #{}:{} |δX|:{:.2e} λ:{:.2e} {}:{:.5f} n:{} dε²:{:.3e} ∇ε²:{:.3e}",
                    out.num_iters, x_str, sqrt(dX_norm2), lambda, e_str, e, nerr, derr, grad_norm2);
      }
      if (!already_rolled_true) {
        x = X_last_good;  // roll back by copy
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
    } else if (options.min_grad_norm2 > 0 && grad_norm2 < options.min_grad_norm2) {
      out.stop_reason = StopReason::kMinGradNorm;
      break;
    }
    // Check duration
    if (options.max_duration_ms > 0 && toc_ms(t) > options.max_duration_ms) {
      out.stop_reason = StopReason::kTimedOut;
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
 * @tparam X_t Type of the parameters to be optimized. Must support arithmetic operations
 * and assignment.
 * @tparam Res_t Type of the residuals function. Must be callable with X_t and
 * return a scalar or a vector of residuals. The function signature is either f(x) or f(x, grad, H)
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
template <typename X_t, typename Res_t>
inline auto Optimize(X_t &x, const Res_t &func, const Options &options = Options{}) {
  if constexpr (std::is_invocable_v<Res_t, const X_t &>) {
    const auto optimize = [](auto &x, const auto &func, const auto &options) {
      return LM(x, func, options);
    };
    return tinyopt::OptimizeJet(x, func, optimize, options);
  } else {
    return LM(x, func, options);
  }
}

}  // namespace tinyopt::lm
