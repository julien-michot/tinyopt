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
#include "tinyopt/math.h"

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
        constexpr bool kIsNLLS = SolverType::IsNLLS;
        return tinyopt::OptimizeWithAutoDiff<kIsNLLS>(x, acc, optimize, options_);
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
    max_iters++;                                // +1 to potentially roll-back
    if (options_.check_final_err) max_iters++;  // one last time to check the final error

    out.errs.reserve(max_iters + 1);
    out.deltas2.reserve(max_iters + 1);
    out.successes.reserve(max_iters + 1);

    // Keep track of the last good 'x'
    constexpr bool kNoCopyX = true;  // TODO offer static alternative to the user
    using BestXType = std::conditional<kNoCopyX, std::nullptr_t, X_t>;
    BestXType *best_x = nullptr;
    if constexpr (!kNoCopyX) best_x = new X_t(x);  // using the copy constructor

    std::optional<Vector<Scalar, Dims>> last_dx;
    bool last_was_success = true;  // Last iteration was a success

    // Run several optimization iterations
    for (int iter = 0; iter < max_iters; ++iter) {
      const auto t = tic();
      const auto &[success, maybe_dx] = Step(x, acc, out);
      bool eval_only = false;

      if (success) {  // Great, let's keep the good work

        ptrait::PlusEq(x, maybe_dx.value());  // Move X by dX
        last_dx = maybe_dx.value();
        last_was_success = true;

        // On the very last iteration, we check that the final error is actually lower
        if (options_.check_final_err && iter + 1 == max_iters) eval_only = true;

      } else {  // Failure to decrease error

        assert(iter != 0);

        if (last_dx) {  // Roll-back
          if constexpr (kNoCopyX)
            ptrait::PlusEq(x, -last_dx.value());  // Move X by -dX
          else
            x = *best_x;
          last_dx.reset();
        } else if (maybe_dx) {  // We failed several times in a row so just evaluate the new x+dx
          ptrait::PlusEq(x, maybe_dx.value());  // Move X by dX
          last_dx = maybe_dx.value();
        }

        eval_only = last_was_success == false;  // No need to build the linear system
        last_was_success = false;
      }

      solver_.Rebuild(!eval_only);

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

    // Copy the very last hessian
    if constexpr (!std::is_null_pointer_v<typename OutputType::H_t>) {
      if (options_.save.H) out.final_H = solver_.Hessian();
    }

    if constexpr (!kNoCopyX) delete best_x;

    if (out.stop_reason == StopReason::kNone && out.num_iters >= max_iters)
      out.stop_reason = StopReason::kMaxIters;
    // Print stop reason
    if (options_.log.enable && out.stop_reason != StopReason::kNone)
      TINYOPT_LOG("{}, final {}:{:.2e}", StopReasonDescription(out, options_), options_.log.e,
                  out.final_err);
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
          out.stop_reason = StopReason::kMaxConsecNoDecr;
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
    const bool has_grad_norm2 =
        options_.min_grad_norm2 > 0.0f || options_.stop_callback || options_.stop_callback2;
    const double grad_norm2 = has_grad_norm2 ? solver_.GradientSquaredNorm() : 0.0;
    if (std::isnan(dx_norm2) || std::isinf(dx_norm2)) {
      if (options_.log.print_failure) {
        TINYOPT_LOG("❌ Failure, dX = \n{}", dx.template cast<float>());
        TINYOPT_LOG("grad = \n{}", solver_.Gradient());
        if constexpr (!SolverType::FirstOrder) TINYOPT_LOG("H = \n{}", solver_.H());
      }
      out.stop_reason = StopReason::kSystemHasNaNOrInf;
      return status;
    }

    // Cost change (negative is good)
    const double derr = err - out.final_err;
    const bool is_good_step = derr < Scalar(0.0);
    // Relative Cost change, defined as (εp-ε)/εp, εp is previous cost,
    const double rel_derr =
        out.final_err > FloatEpsilon<Scalar>() && out.final_err < std::numeric_limits<Scalar>::max()
            ? (out.final_err - err) / out.final_err
            : 0.0f;
    // Save history of errors and deltas
    out.errs.emplace_back(err);
    out.deltas2.emplace_back(dx_norm2);
    out.successes.emplace_back(is_good_step);

    // Log
    if (options_.log.enable) {
      std::ostringstream oss;
      if (options_.log.print_emoji) oss << (is_good_step ? (iter == 0 ? "ℹ️" : "✅") : "❌");
      oss << "#" << iter << " ";
      if (options_.log.print_t) {
        oss << TINYOPT_FORMAT_NS::format("τ:{:.2f} ", out.duration_ms);
      }
      if (options_.log.print_x) {
        if constexpr (traits::is_scalar_v<X_t>) {
          oss << TINYOPT_FORMAT_NS::format("x:{:.5f} ", x);
        } else if constexpr (traits::is_matrix_or_array_v<X_t>) {  // Flattened X
          oss << "x:["
#ifdef TINYOPT_NO_FORMATTERS
              << x.reshaped().transpose()
#else
              << TINYOPT_FORMAT_NS::format("{} ", x.reshaped().transpose())
#endif  // TINYOPT_NO_FORMATTERS
              << "] ";
        } else if constexpr (traits::is_streamable_v<X_t>) {
          // User must define the stream operator of ParameterType
          oss << "{" << x << "} ";
        }
      }
      // Print step info
      oss << TINYOPT_FORMAT_NS::format("|δx|:{:.2e} ", sqrt(dx_norm2));
      // Estimate max standard deviations from (co)variances
      if constexpr (!SolverType::FirstOrder) {
        if (is_good_step && options_.log.print_max_stdev)
          oss << TINYOPT_FORMAT_NS::format("⎡σ⎤:{:.2f} ", solver_.MaxStdDev());
      }
      // Print error
      oss << TINYOPT_FORMAT_NS::format("{}:{:.4e} n:{} d{}:{:+.2e} r{}:{:+.1e} ", options_.log.e,
                                       err, nerr, options_.log.e, derr, options_.log.e, rel_derr);
      if (has_grad_norm2) oss << TINYOPT_FORMAT_NS::format("|∇|:{:.2e} ", sqrt(grad_norm2));
      oss << solver_.stateAsString();
      TINYOPT_LOG("{}", oss.str());
    }

    // Update output struct
    if (is_good_step) { /* GOOD Step */
      // Note: we guess it's a good step in the first iteration
      solver_.GoodStep(options_.use_step_quality_approx ? rel_derr : 0.0f);
      out.num_consec_failures = 0;
      out.final_err = err;
      out.final_rerr_dec = rel_derr;
    } else { /* BAD Step */
      solver_.BadStep();
      out.num_failures++;
      out.num_consec_failures++;
      if (options_.max_consec_failures > 0 &&
          out.num_consec_failures >= options_.max_consec_failures) {
        out.stop_reason = StopReason::kMaxConsecNoDecr;
        return status;
      }
      if (options_.max_total_failures > 0 && out.num_failures >= options_.max_total_failures) {
        out.stop_reason = StopReason::kMaxNoDecr;
        return status;
      }
    }

    // Detect if we need to stop
    if (solver_failed)
      out.stop_reason = StopReason::kSolverFailed;
    else if (options_.min_error > 0 && err < options_.min_error)
      out.stop_reason = StopReason::kMinError;
    else if (options_.min_rerr_dec > 0 && rel_derr > 0.0 && rel_derr < options_.min_rerr_dec)
      out.stop_reason = StopReason::kMinRelError;
    else if (options_.min_step_norm2 > 0 && dx_norm2 < options_.min_step_norm2)
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
