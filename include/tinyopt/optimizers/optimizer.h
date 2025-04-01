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

#include <cstddef>
#include <cstdint>
#include <optional>
#include <variant>

#include <tinyopt/log.h>
#include <tinyopt/opt_jet.h>
#include <tinyopt/options.h>
#include <tinyopt/output.h>
#include <tinyopt/time.h>
#include <tinyopt/traits.h>

namespace tinyopt::optimizers {

/***
 *  @brief Optimizer
 */
template <typename SolverType, typename _Options = tinyopt::CommonOptions>
class Optimizer {
 public:
  using Scalar = typename SolverType::Scalar;
  static constexpr int Dims = SolverType::Dims;
  static constexpr bool FirstOrder = SolverType::FirstOrder;
  using OutputType = std::conditional_t<SolverType::FirstOrder, Output<std::nullptr_t>,
                                        Output<typename SolverType::H_t>>;

  struct Options : _Options {
    Options(const _Options &options_ = {},
            const SolverType::Options solver_options = {})
        : _Options{options_}, solver{solver_options} {}

    /// Solver options
    SolverType::Options solver;
  };

  Optimizer(const Options &_options = {}) : options_{_options}, solver_(_options.solver) {}

  /// Initialize solver with specific gradient and hessian
  template <int FO = FirstOrder, std::enable_if_t<!FO, int> = 0>
  void InitWith(const auto &g, const auto &h) {
    solver_.InitWith(g, h);
  }

  /// Initialize solver with specific gradient
  template <int FO = FirstOrder, std::enable_if_t<FO, int> = 0>
  void InitWith(const auto &g) {
    solver_.InitWith(g);
  }

  /// Reset the optimization and solver
  void reset() { solver_.reset(); }

  /// Main optimization function
  template <typename X_t, typename AccFunc>
  OutputType operator()(X_t &x, const AccFunc &acc, int num_iters = -1) {
    // Detect if we need to do automatic differentiation
    if constexpr (std::is_invocable_v<AccFunc, const X_t &>) {
      const auto optimize = [&](auto &x, const auto &func, const auto &) {
        return Optimize(x, func, num_iters);
      };
      return tinyopt::OptimizeJet(x, acc, optimize, options_);
    } else {  // AD not needed
      return Optimize(x, acc, num_iters);
    }
  }

  template <typename X_t>
  std::variant<StopReason, bool> ResizeIfNeeded(X_t &x) {
    using ptrait = traits::params_trait<X_t>;
    int dims = Dims;  // Dynamic size
    if constexpr (Dims == Dynamic) dims = ptrait::dims(x);
    if (Dims == Dynamic || dims == 0) {
      TINYOPT_LOG("Error: Parameters dimensions cannot be 0 or Dynamic at execution time");
      return StopReason::kSkipped;
    }

    // Resize the solver if needed TODO move?
    bool resized = false;
    try {
      resized = solver_.resize(dims);
      if constexpr (!std::is_same_v<typename OutputType::H_t, std::nullptr_t>) {
        // TODO if (options_.export_H) out.last_H.setZero();
      }
    } catch (const std::bad_alloc &e) {
      if (options_.log.enable) {
        TINYOPT_LOG(
            "Failed to allocate {} Hessian(s) of size {}x{}, "
            "mem:{}GB, maybe use a SparseMatrix?",
            options_.export_H ? 2 : 1, dims, dims, 1e-9 * dims * dims * sizeof(Scalar));
      }
      return StopReason::kOutOfMemory;
    } catch (const std::invalid_argument &e) {
      TINYOPT_LOG("Error: Failed to resize solver");
      return StopReason::kSkipped;
    }
    return resized;
  }

  /// Run one optimization iteration, return the stopping criteria that are met, if any
  template <typename X_t, typename AccFunc>
  std::optional<StopReason> Step(X_t &x, const AccFunc &acc, OutputType &out) {
    using ptrait = traits::params_trait<X_t>;
    const auto num_iters = out.num_iters;
    out.num_iters++;

    int dims = Dims;  // Dynamic size
    if constexpr (Dims == Dynamic) dims = ptrait::dims(x);

    // Resize the solver if needed
    const auto resize_status = ResizeIfNeeded(x);
    if (auto fail_reason = std::get_if<StopReason>(&resize_status)) {
      out.stop_reason = *fail_reason;
      return *fail_reason;
    }

    solver_.clear();  // TODO try to remove

    const std::string e_str = options_.log.print_rmse ? "√ε²/n" : "ε²";

    bool already_rolled_true = true;
    const uint8_t max_tries =
        options_.max_consec_failures > 0 ? std::max<uint8_t>(1, options_.max_total_failures) : 255;
    auto X_last_good = x;

    // Create the gradient and displacement `dx`
    Vector<Scalar, Dims> dx(dims);
    double err = 0;  // accumulated error (for monotony check and logging)
    int nerr = 0;    // number of residuals (optional, for logging)

    bool solver_failed = true;
    // Solver linear a few times until it's enough
    for (; out.num_consec_failures <= max_tries;) {
      // Solver for dx
      bool success = solver_.Solve(x, acc, dx);
      err = solver_.Error();
      nerr = solver_.NumResiduals();
      // Check success/failure
      if (success) {
        solver_failed = false;
        break;
      } else {  // Failure
        out.num_consec_failures++;
        out.num_failures++;
        // Check there's some residuals
        if (nerr == 0) {
          if (options_.log.enable) TINYOPT_LOG("❌ #{}: No residuals, stopping", num_iters);
          out.stop_reason = StopReason::kSkipped;
          return out.stop_reason;
        } else {
          if (options_.log.enable)
            TINYOPT_LOG("❌ #{}:Failed to solve the linear system", num_iters);
          solver_.Failed(10);  // Tell the solver it's a failure... and try again
        }
      }
    }

    // Check for NaNs and Inf
    if (std::isnan(err) || std::isinf(err)) {
      if (options_.log.enable) TINYOPT_LOG("❌ #{}: NaN/Inf in error", num_iters);
      // Can break only if first time, otherwise better count it as failure
      out.stop_reason = StopReason::kSystemHasNaNOrInf;
      return out.stop_reason;
    }

    // Check the displacement magnitude
    const double dX_norm2 = solver_failed ? 0 : dx.squaredNorm();
    const double grad_norm2 = options_.min_grad_norm2 == 0.0f ? 0 : solver_.GradientSquaredNorm();
    if (std::isnan(dX_norm2) || std::isinf(dX_norm2)) {
      solver_failed = true;
      if (options_.log.print_failure) {
        TINYOPT_LOG("❌ Failure, dX = \n{}", dx.template cast<float>());
        TINYOPT_LOG("H = \n{}", solver_.H());
        TINYOPT_LOG("grad = \n{}", solver_.Gradient());
      }
      out.stop_reason = StopReason::kSystemHasNaNOrInf;
      return out.stop_reason;
    }

    const double derr = err - out.last_err2;
    // Save history of errors and deltas
    out.errs2.emplace_back(err);
    out.deltas2.emplace_back(dX_norm2);
    // Convert X to string (if log enabled)
    std::string x_str;
    if (options_.log.print_x) {
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
      if (num_iters > 0) X_last_good = x;
      // Move X by dX
      ptrait::pluseq(x, dx);
      // Save results
      out.last_err2 = err;
      if constexpr (!std::is_same_v<typename OutputType::H_t, std::nullptr_t>) {
        if (options_.export_H) out.last_H = solver_.Hessian();
      }
      already_rolled_true = false;
      out.num_consec_failures = 0;
      // Log
      if (options_.log.enable) {
        const double e = options_.log.print_rmse ? std::sqrt(err / nerr) : err;
        // Estimate max standard deviations from (co)variances
        std::ostringstream oss_sigma;
        if constexpr (!SolverType::FirstOrder) {
          if (options_.log.print_max_stdev) oss_sigma << "⎡σ⎤:" << solver_.MaxStdDev() << " ";
        }
        TINYOPT_LOG("✅ #{}:{} |δX|:{:.2e} {}{}{}:{:.5f} n:{} dε²:{:.3e} ∇ε²:{:.3e}", num_iters,
                    x_str, sqrt(dX_norm2), solver_.LogString(), oss_sigma.str(), e_str, e, nerr,
                    derr, grad_norm2);
      }

      solver_.Succeeded();
    } else { /* BAD Step */
      out.successes.emplace_back(false);
      // Log
      if (options_.log.enable) {
        const double e = options_.log.print_rmse ? std::sqrt(err / nerr) : err;
        TINYOPT_LOG("❌ #{}:{} |δX|:{:.2e} {}{}:{:.5f} n:{} dε²:{:.3e} ∇ε²:{:.3e}", num_iters,
                    x_str, sqrt(dX_norm2), solver_.LogString(), e_str, e, nerr, derr, grad_norm2);
      }
      if (!already_rolled_true) {
        x = X_last_good;  // roll back by copy
        already_rolled_true = true;
      }
      out.num_failures++;
      out.num_consec_failures++;
      if (options_.max_consec_failures > 0 &&
          out.num_consec_failures >= options_.max_consec_failures) {
        out.stop_reason = StopReason::kMaxConsecFails;
        return out.stop_reason;
      }
      if (options_.max_total_failures > 0 && out.num_failures >= options_.max_total_failures) {
        out.stop_reason = StopReason::kMaxFails;
        return out.stop_reason;
      }
      solver_.Failed();
    }

    // Detect if we need to stop and the reason
    if (solver_failed) {
      out.stop_reason = StopReason::kSolverFailed;
      return out.stop_reason;
    } else if (options_.min_delta_norm2 > 0 && dX_norm2 < options_.min_delta_norm2) {
      out.stop_reason = StopReason::kMinDeltaNorm;
      return out.stop_reason;
    } else if (options_.min_grad_norm2 > 0 && grad_norm2 < options_.min_grad_norm2) {
      out.stop_reason = StopReason::kMinGradNorm;
      return out.stop_reason;
    }
    out.stop_reason = StopReason::kNone;
    return std::nullopt;
  }

 protected:
  /// Main optimization loop
  template <typename X_t, typename AccFunc>
  OutputType Optimize(X_t &x, const AccFunc &acc, int num_iters = -1) {
    OutputType out;
    if (num_iters < 0) num_iters = options_.num_iters;
    const auto t = tic();

    out.errs2.reserve(num_iters + 1);
    out.deltas2.reserve(num_iters + 1);
    out.successes.reserve(num_iters + 1);

    // Run several optimization iterations
    while (out.num_iters < num_iters + 1 /*+1 to potentially roll-back*/) {
      auto stop = Step(x, acc, out);  // increment out.num_iters
      if (stop) {
        out.stop_reason = stop.value();
        break;
      }
      // Check whether it's taking too much time
      if (options_.max_duration_ms > 0 && toc_ms(t) > options_.max_duration_ms) {
        out.stop_reason = StopReason::kTimedOut;
        break;
      }
    }
    return out;
  }

 protected:
  /// Optimization options
  const Options options_;
  /// Linear solver
  SolverType solver_;
};

}  // namespace tinyopt::optimizers
