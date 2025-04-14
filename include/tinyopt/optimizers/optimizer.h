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

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <variant>

#include <tinyopt/log.h>
#include <tinyopt/output.h>
#include <tinyopt/time.h>
#include <tinyopt/traits.h>

#include <tinyopt/optimizers/options.h>

#ifndef TINYOPT_DISABLE_AUTODIFF
#include <tinyopt/diff/optimize_autodiff.h>
#endif
#ifndef TINYOPT_DISABLE_NUMDIFF
#include <tinyopt/diff/num_diff.h>
#endif

namespace tinyopt::optimizers {

/***
 *  @brief Optimizer
 */
template <typename SolverType, typename _Options = std::nullptr_t>
class Optimizer {
 public:
  using Scalar = typename SolverType::Scalar;
  static constexpr Index Dims = SolverType::Dims;
  using OutputType = Output<typename SolverType::H_t>;

 private:
  /// Default Options struct in case `_Options` is a nullptr_t
  struct DefaultOptions : Options2 {
    DefaultOptions(const Options2 options = {}) : Options2{options} {}
    SolverType::Options solver;
  };

 public:
  using Options = std::conditional_t<std::is_null_pointer_v<_Options>, DefaultOptions, _Options>;

  Optimizer(const Options &_options = {}) : options_{_options}, solver_(_options.solver) {}

  /// Initialize solver with specific gradient and hessian
  template <int FO = SolverType::FirstOrder, std::enable_if_t<!FO, int> = 0>
  void InitWith(const auto &g, const auto &h) {
    solver_.InitWith(g, h);
  }

  /// Initialize solver with specific gradient
  template <int FO = SolverType::FirstOrder, std::enable_if_t<FO, int> = 0>
  void InitWith(const auto &g) {
    solver_.InitWith(g);
  }

  /// Reset the optimization and solver
  void reset() { solver_.reset(); }

  template <typename X_t>
  std::variant<StopReason, bool> ResizeIfNeeded(X_t &x, OutputType &out) {
    using ptrait = traits::params_trait<X_t>;
    Index dims = Dims;  // Dynamic size
    if constexpr (Dims == Dynamic) dims = ptrait::dims(x);
    if (Dims == Dynamic && dims == 0) {
      TINYOPT_LOG("Error: Parameters dimensions cannot be 0 or Dynamic at execution time");
      return StopReason::kSkipped;
    }

    // Resize the solver if needed TODO move?
    bool resized = false;
    try {
      resized = solver_.resize(dims);
      if constexpr (std::is_base_of_v<typename SolverType::Options, Options2>)
        if (options_.save.H) out.final_H.setZero();
    } catch (const std::bad_alloc &) {
      if (options_.log.enable) {
        int num_hessians = 1;
        if constexpr (std::is_base_of_v<typename SolverType::Options, Options2>)
          if (options_.save.H) num_hessians++;
        TINYOPT_LOG(
            "Failed to allocate {} Hessian(s) of size {}x{}, "
            "mem:{}GB, maybe use a SparseMatrix?",
            num_hessians, dims, dims, 1e-9f * static_cast<float>(dims * dims * sizeof(Scalar)));
      }
      return StopReason::kOutOfMemory;
    } catch (const std::invalid_argument &e) {
      TINYOPT_LOG("Error: Failed to resize the linear solver. {}", e.what());
      return StopReason::kSkipped;
    }
    return resized;
  }

  /// Main optimization function
  template <typename X_t, typename AccFunc>
  OutputType operator()(X_t &x, const AccFunc &acc, int max_iters = -1) {
    // Detect if we need to do  differentiation
    if constexpr (std::is_invocable_v<AccFunc, const X_t &>) {
      // Try to run AD
#ifndef TINYOPT_DISABLE_AUTODIFF
      using Jet = diff::Jet<Scalar, Dims>;
      using XJetType =
          std::conditional_t<std::is_floating_point_v<X_t>, Jet,
                             decltype(traits::params_trait<X_t>::template cast<Jet>(x))>;
      if constexpr (std::is_invocable_v<AccFunc, const XJetType &>) {
        const auto optimize = [&](auto &x, const auto &func, const auto &) {
          return Optimize(x, func, max_iters);
        };
        return tinyopt::OptimizeWithAutoDiff(x, acc, optimize, options_);
      }
#else
      if constexpr (0) {
      }
#endif  // TINYOPT_DISABLE_AUTODIFF

#ifndef TINYOPT_DISABLE_NUMDIFF
      else {
        // Add warning at compilation
        // TODO #pragma message("Your function cannot be auto-differentiated, using numerical
        // differentiation")
        if constexpr (SolverType::FirstOrder) {
          auto loss = diff::CreateNumDiffFunc1(x, acc);
          return Optimize(x, loss, max_iters);
        } else {
          auto loss = diff::CreateNumDiffFunc2(x, acc);
          return Optimize(x, loss, max_iters);
        }
      }
#else
      else {
        static_assert(false, "Cannot do differentiation...");
      }
#endif  // TINYOPT_DISABLE_NUMDIFF
    } else {
      return Optimize(x, acc, max_iters);
    }
  }

  /// Main optimization loop
  template <typename X_t, typename AccFunc>
  OutputType Optimize(X_t &x, const AccFunc &acc, int max_iters = -1) {
    using ptrait = traits::params_trait<X_t>;
    OutputType out;
    // Set start time
    out.start_time = tic();
    if (max_iters < 0) max_iters = options_.max_iters;

    out.errs.reserve(max_iters + 1);
    out.deltas2.reserve(max_iters + 1);
    out.successes.reserve(max_iters + 1);

    // Keep track of the last good 'x'
    constexpr bool kNoCopyX = true;  // TODO offer static alternative to the user
    using BestXType = std::conditional<kNoCopyX, std::nullptr_t, X_t>;
    BestXType *best_x = nullptr;
    if constexpr (!kNoCopyX) best_x = new X_t(x);  // using the copy constructor

    std::optional<Vector<Scalar, Dims>> last_dx;

    // Run several optimization iterations
    for (int iter = 0; iter < max_iters + 1 /*+1 to potentially roll-back*/; ++iter) {
      const auto t = tic();
      const auto &[success, maybe_dx] = Step(x, acc, out);

      if (!success && last_dx) {  // Roll-back 'x' to 'best_x' and throw away the step dx
        assert(iter != 0);
        if constexpr (kNoCopyX)
          ptrait::PlusEq(x, -last_dx.value());  // Move X by -dX
        else
          x = *best_x;
        last_dx.reset();
      } else if (maybe_dx.has_value()) {      // Accept the step and verify it at the next iteration
        ptrait::PlusEq(x, maybe_dx.value());  // Move X by dX
        last_dx = maybe_dx.value();
      }

      // Check for a time out
      out.duration_ms += static_cast<float>(toc_ms(t));
      if (options_.max_duration_ms > 0 && out.duration_ms > options_.max_duration_ms) {
        out.stop_reason = StopReason::kTimedOut;
      }
      // Iteration done
      out.num_iters++;
      // Stop now?
      if (out.stop_reason != StopReason::kNone) break;
    }

    // On the very last iteration, we check that the final error is actually lower
    if (options_.check_final_err && last_dx) {
      const auto err = solver_.Evaluate(x, acc);
      if (err > out.final_err) {
        if (options_.log.enable)
          TINYOPT_LOG("ℹ️ Re-evaluated error {:.2e} > {:.2e} (before), rolling back.", err,
                      out.final_err);
        if constexpr (kNoCopyX)
          ptrait::PlusEq(x, -last_dx.value());  // Move X by -dX
        else
          x = *best_x;
      }
    }

    if constexpr (!kNoCopyX) delete best_x;

    // Print stop reason
    if (options_.log.enable && out.stop_reason != StopReason::kNone) {
      TINYOPT_LOG("{}", out.StopReasonDescription());
    }
    return out;
  }

 protected:
  /// Run one optimization iteration, return the estimated next step (solve + decreased error)
  template <typename X_t, typename AccFunc>
  std::pair<bool, std::optional<Vector<Scalar, Dims>>> Step(X_t &x, const AccFunc &acc,
                                                            OutputType &out) {
    const auto iter = out.num_iters;
    std::pair<bool, std::optional<Vector<Scalar, Dims>>> status{false, std::nullopt};

    // Set start time if not set already
    const auto t = tic();
    if (out.start_time == TimePoint::min()) out.start_time = t;

    // Resize the solver if needed
    const auto resize_status = ResizeIfNeeded(x, out);
    if (auto fail_reason = std::get_if<StopReason>(&resize_status)) {
      out.stop_reason = *fail_reason;
      return status;
    }

    const bool resize_and_clear_solver = true;  // for now
    const std::string e_str = options_.log.e;   // error symbol, default is 'ε'

    // Create the gradient and displacement `dx`
    Vector<Scalar, Dims> dx;
    Scalar err = NAN;              // accumulated error (for monotony check and logging)
    int nerr = out.num_residuals;  // number of residuals (optional, for logging)

    bool solver_failed = true;
    // Solver linear a few times until it's enough
    const uint8_t max_tries =
        options_.max_consec_failures > 0 ? std::max<uint8_t>(1, options_.max_consec_failures) : 255;
    for (; out.num_consec_failures <= max_tries;) {
      // Accumulate residuals and jacobians
      if (solver_.Build(x, acc, resize_and_clear_solver)) {
        // Ok, let's try to solve for `dx` now
        if (const auto &maybe_dx = solver_.Solve()) {
          dx = maybe_dx.value();  // TODO void copy?
          solver_failed = false;
        } else {
        }
      }
      // Check success/failure
      if (!solver_failed) {
        err = solver_.Error();
        nerr = solver_.NumResiduals();
        solver_failed = false;
        break;
      } else {  // Failure
        out.num_consec_failures++;
        out.num_failures++;
        // Check there's some residuals
        if (nerr == 0) {
          if (options_.log.enable) TINYOPT_LOG("❌ #{}: No residuals, stopping", iter);
          out.stop_reason = StopReason::kSkipped;
          return status;
        } else if (options_.max_consec_failures > 0 &&
                   out.num_consec_failures >= options_.max_consec_failures) {
          out.stop_reason = StopReason::kMaxConsecFails;
          return status;
        } else if (options_.log.enable)
          TINYOPT_LOG("❌ #{}:Failed to solve the linear system", iter);
        solver_.FailedStep();  // Tell the solver it's a failure... and try again
      }
    }

    // Check for NaNs and Inf
    if (std::isnan(err) || std::isinf(err)) {
      if (options_.log.enable) TINYOPT_LOG("❌ #{}: NaN/Inf in error", iter);
      // Can break only if first time, otherwise better count it as failure
      out.stop_reason = StopReason::kSystemHasNaNOrInf;
      return status;
    }

    // Stop here if the solver failed constantly
    if (solver_failed) {
      out.stop_reason = StopReason::kSolverFailed;
      return status;
    }

    // Check the displacement magnitude
    const double dx_norm2 = solver_failed ? 0 : dx.squaredNorm();
    const double grad_norm2 =
        (options_.min_grad_norm2 == 0.0f || options_.stop_callback || options_.stop_callback2)
            ? 0
            : solver_.GradientSquaredNorm();
    if (std::isnan(dx_norm2) || std::isinf(dx_norm2)) {
      if (options_.log.print_failure) {
        TINYOPT_LOG("❌ Failure, dX = \n{}", dx.template cast<float>());
        TINYOPT_LOG("grad = \n{}", solver_.Gradient());
        if constexpr (!SolverType::FirstOrder) TINYOPT_LOG("H = \n{}", solver_.H());
      }
      out.stop_reason = StopReason::kSystemHasNaNOrInf;
      return status;
    }

    const double derr = err - out.final_err;
    // Save history of errors and deltas
    out.errs.emplace_back(err);
    out.deltas2.emplace_back(dx_norm2);
    // Convert X to string (if log enabled)
    std::ostringstream prefix_oss, oss_sigma;
    if (options_.log.enable) {
      // Adding iters
      prefix_oss << "#" << iter << ":";
      if (options_.log.print_t) {
        prefix_oss << TINYOPT_FORMAT_NAMESPACE::format(" τ:{:.2f}ms", out.duration_ms);
      }
      if (options_.log.print_x) {
        if constexpr (traits::is_matrix_or_array_v<X_t>) {  // Flattened X
          prefix_oss << " x:["
#ifdef TINYOPT_NO_FORMATTERS
                     << x.reshaped().transpose();
#else
                     << TINYOPT_FORMAT_NAMESPACE::format("{}", x.reshaped().transpose());
#endif  // TINYOPT_NO_FORMATTERS
          prefix_oss << "]";
        } else if constexpr (traits::is_streamable_v<X_t>) {
          // User must define the stream operator of ParameterType
          prefix_oss << " x:{" << x << "}";
        }
      }
    }

    // Update x += dx and eventually check the error
    bool is_good_step = derr < Scalar(0.0);
    if (is_good_step) { /* GOOD Step */
      if constexpr (!std::is_null_pointer_v<typename OutputType::H_t>) {
        if (options_.save.H) out.final_H = solver_.Hessian();
      }
      out.successes.emplace_back(true);
      out.num_consec_failures = 0;
      // Estimate the relative error decrease
      const Scalar step_quality = out.final_err >= std::numeric_limits<Scalar>::max()
                                      ? 0.0f
                                      : solver_.EstimateStepQuality(dx);
      const Scalar rel_derr = step_quality > FloatEpsilon<Scalar>() ? derr / step_quality : 0.0f;
      solver_.GoodStep(rel_derr);
      out.final_err = err;
    } else { /* BAD Step */
      out.successes.emplace_back(false);
      out.num_failures++;
      out.num_consec_failures++;
      if (options_.max_consec_failures > 0 &&
          out.num_consec_failures >= options_.max_consec_failures) {
        out.stop_reason = StopReason::kMaxConsecFails;
        return status;
      }
      if (options_.max_total_failures > 0 && out.num_failures >= options_.max_total_failures) {
        out.stop_reason = StopReason::kMaxFails;
        return status;
      }
      solver_.BadStep();
    }
    // Log
    if (options_.log.enable) {
      // Estimate max standard deviations from (co)variances
      if constexpr (!SolverType::FirstOrder) {
        if (is_good_step && options_.log.print_max_stdev)
          oss_sigma << TINYOPT_FORMAT_NAMESPACE::format("⎡σ⎤:{:.2f} ", solver_.MaxStdDev());
      }
      TINYOPT_LOG("{} {} |δx|:{:.2e} {}{}{}:{:.2e} n:{} dε:{:.3e} |∇|:{:.3e}",
                  is_good_step ? (iter == 0 ? "ℹ️" : "✅") : "❌", prefix_oss.str(), sqrt(dx_norm2),
                  solver_.stateAsString(), oss_sigma.str(), e_str, err, nerr, derr, grad_norm2);
    }

    // Detect if we need to stop
    if (solver_failed)
      out.stop_reason = StopReason::kSolverFailed;
    else if (options_.min_error > 0 && err < options_.min_error)
      out.stop_reason = StopReason::kMinError;
    else if (options_.min_delta_norm2 > 0 && dx_norm2 < options_.min_delta_norm2)
      out.stop_reason = StopReason::kMinDeltaNorm;
    else if (options_.min_grad_norm2 > 0 && grad_norm2 < options_.min_grad_norm2)
      out.stop_reason = StopReason::kMinGradNorm;
    else if (options_.stop_callback && options_.stop_callback(err, dx_norm2, grad_norm2))
      out.stop_reason = StopReason::kUserStopped;
    else if (options_.stop_callback2 &&
             options_.stop_callback2(float(err), dx.template cast<float>(),
                                     solver_.Gradient().template cast<float>()))
      out.stop_reason = StopReason::kUserStopped;

    status.first = is_good_step;
    status.second = dx;
    return status;
  }

 protected:
  /// Optimization options
  const Options options_;
  /// Linear solver
  SolverType solver_;
};

}  // namespace tinyopt::optimizers
