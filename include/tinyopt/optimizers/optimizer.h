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
#include <type_traits>
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
  static constexpr int Dims = SolverType::Dims;
  using OutputType = Output<typename SolverType::H_t>;

 private:
  /// Default Options struct in case `_Options` is a nullptr_t
  struct DefaultOptions : Options2 {
    DefaultOptions(const Options2 options = {}) : Options2{options} {}
    SolverType::Options solver;
  };

 public:
  using Options =
      std::conditional_t<std::is_same_v<_Options, std::nullptr_t>, DefaultOptions, _Options>;

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

  /// Main optimization function
  template <typename X_t, typename AccFunc>
  OutputType operator()(X_t &x, const AccFunc &acc, int num_iters = -1) {
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
          return Optimize(x, func, num_iters);
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
          return Optimize(x, loss, num_iters);
        } else {
          auto loss = diff::CreateNumDiffFunc2(x, acc);
          return Optimize(x, loss, num_iters);
        }
      }
#else
      else {
        static_assert(false, "Cannot do differentiation...");
      }
#endif  // TINYOPT_DISABLE_NUMDIFF
    } else {
      return Optimize(x, acc, num_iters);
    }
  }

  /// Main optimization loop
  template <typename X_t, typename AccFunc>
  OutputType Optimize(X_t &x, const AccFunc &acc, int num_iters = -1) {
    OutputType out;
    if (num_iters < 0) num_iters = options_.num_iters;

    out.errs.reserve(num_iters + 1);
    out.deltas2.reserve(num_iters + 1);
    out.successes.reserve(num_iters + 1);

    // Run several optimization iterations
    for (int iter = 0; iter <= num_iters + 1 /*+1 to potentially roll-back*/; ++iter) {
      const auto stop = Step(x, acc, out);  // increment out.num_iters
      if (stop) break;
    }
    return out;
  }

  template <typename X_t>
  std::variant<StopReason, bool> ResizeIfNeeded(X_t &x, OutputType &out) {
    using ptrait = traits::params_trait<X_t>;
    int dims = Dims;  // Dynamic size
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
        if (options_.save.H) out.last_H.setZero();
    } catch (const std::bad_alloc &e) {
      if (options_.log.enable) {
        int num_hessians = 1;
        if constexpr (std::is_base_of_v<typename SolverType::Options, Options2>)
          if (options_.save.H) num_hessians++;
        TINYOPT_LOG(
            "Failed to allocate {} Hessian(s) of size {}x{}, "
            "mem:{}GB, maybe use a SparseMatrix?",
            num_hessians, dims, dims, 1e-9 * dims * dims * sizeof(Scalar));
      }
      return StopReason::kOutOfMemory;
    } catch (const std::invalid_argument &e) {
      TINYOPT_LOG("Error: Failed to resize the linear solver");
      return StopReason::kSkipped;
    }
    return resized;
  }

  /// Run one optimization iteration, return the stopping criteria that are met, if any
  template <typename X_t, typename AccFunc>
  std::optional<StopReason> Step(X_t &x, const AccFunc &acc, OutputType &out) {
    using ptrait = traits::params_trait<X_t>;
    const auto num_iters = out.num_iters;

    // Set start time if not set already
    const auto t = tic();
    if (out.start_time == TimePoint::min()) out.start_time = t;

    // Resize the solver if needed
    const auto resize_status = ResizeIfNeeded(x, out);
    if (auto fail_reason = std::get_if<StopReason>(&resize_status)) {
      out.stop_reason = *fail_reason;
      return *fail_reason;
    }

    const bool resize_and_clear_solver = true;  // for now

    const std::string e_str = options_.log.print_mean_x ? "ε/n" : "ε";

    bool already_rolled_true = true;
    const uint8_t max_tries =
        options_.max_consec_failures > 0 ? std::max<uint8_t>(1, options_.max_total_failures) : 255;
    auto X_last_good = x;

    // Create the gradient and displacement `dx`
    Vector<Scalar, Dims> dx;
    double err = out.last_err;     // accumulated error (for monotony check and logging)
    int nerr = out.num_residuals;  // number of residuals (optional, for logging)

    bool solver_failed = true;
    // Solver linear a few times until it's enough
    for (; out.num_consec_failures <= max_tries;) {
      // Accumulate residuals and jacobians
      bool success = false;
      if (solver_.Build(x, acc, resize_and_clear_solver)) {
        // Ok, let's try to solve for `dx` now
        if (const auto &maybe_dx = solver_.Solve()) {
          dx = maybe_dx.value();  // TODO void copy?
          success = true;
        }
      }
      // Check success/failure
      if (success) {
        err = solver_.Error();
        nerr = solver_.NumResiduals();
        solver_failed = false;
        break;
      } else {                      // Failure
        out.num_consec_failures++;  // TODO add a max_num_failures to solve instead
        out.num_failures++;
        // Check there's some residuals
        if (nerr == 0) {
          if (options_.log.enable) TINYOPT_LOG("❌ #{}: No residuals, stopping", num_iters);
          out.stop_reason = StopReason::kSkipped;
          goto closure;
        } else if (options_.max_consec_failures > 0 &&
                   out.num_consec_failures >= options_.max_consec_failures) {
          out.stop_reason = StopReason::kMaxConsecFails;
          goto closure;
        } else if (options_.log.enable)
          TINYOPT_LOG("❌ #{}:Failed to solve the linear system", num_iters);
        solver_.Failed(10);  // Tell the solver it's a failure... and try again
      }
    }

    // Check for NaNs and Inf
    if (std::isnan(err) || std::isinf(err)) {
      if (options_.log.enable) TINYOPT_LOG("❌ #{}: NaN/Inf in error", num_iters);
      // Can break only if first time, otherwise better count it as failure
      out.stop_reason = StopReason::kSystemHasNaNOrInf;
      goto closure;
    }

    {
      // Check the displacement magnitude
      const double dX_norm2 = solver_failed ? 0 : dx.squaredNorm();
      const double grad_norm2 = (options_.min_grad_norm2 == 0.0f || options_.stop_callback)
                                    ? 0
                                    : solver_.GradientSquaredNorm();
      if (std::isnan(dX_norm2) || std::isinf(dX_norm2)) {
        solver_failed = true;
        if (options_.log.print_failure) {
          TINYOPT_LOG("❌ Failure, dX = \n{}", dx.template cast<float>());
          TINYOPT_LOG("grad = \n{}", solver_.Gradient());
          if constexpr (!SolverType::FirstOrder) TINYOPT_LOG("H = \n{}", solver_.H());
        }
        out.stop_reason = StopReason::kSystemHasNaNOrInf;
        goto closure;
      }

      const double derr = err - out.last_err;
      // Save history of errors and deltas
      out.errs.emplace_back(err);
      out.deltas2.emplace_back(dX_norm2);
      // Convert X to string (if log enabled)
      std::ostringstream prefix_oss;
      // Adding iters
      prefix_oss << "#" << num_iters << ":";
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
      // Check step quality
      if (derr < 0.0 && !solver_failed) { /* GOOD Step */
        out.successes.emplace_back(true);
        if (num_iters > 0) X_last_good = x;
        // Move X by dX
        ptrait::pluseq(x, dx);
        // Save results
        out.last_err = err;
        if constexpr (!std::is_null_pointer_v<typename OutputType::H_t>) {
          if (options_.save.H) out.last_H = solver_.Hessian();
          if (options_.save.acc_dx) out.last_acc_dx += dx;
        }
        already_rolled_true = false;
        out.num_consec_failures = 0;
        // Log
        if (options_.log.enable) {
          const double e = options_.log.print_mean_x ? std::sqrt(err / nerr) : err;
          // Estimate max standard deviations from (co)variances
          std::ostringstream oss_sigma;
          if constexpr (!SolverType::FirstOrder) {
            if (options_.log.print_max_stdev)
              oss_sigma << TINYOPT_FORMAT_NAMESPACE::format("⎡σ⎤:{:.2f} ", solver_.MaxStdDev());
          }
          TINYOPT_LOG("✅ {} |δx|:{:.2e} {}{}{}:{:.2e} n:{} dε:{:.3e} |∇|:{:.3e}", prefix_oss.str(),
                      sqrt(dX_norm2), solver_.LogString(), oss_sigma.str(), e_str, e, nerr, derr,
                      grad_norm2);
        }

        solver_.Succeeded();
      } else { /* BAD Step */
        out.successes.emplace_back(false);
        // Log
        if (options_.log.enable) {
          const double e = options_.log.print_mean_x ? std::sqrt(err / nerr) : err;
          TINYOPT_LOG("❌ {} |δx|:{:.2e} {}{}:{:.2e} n:{} dε:{:.3e} |∇|:{:.3e}", prefix_oss.str(),
                      sqrt(dX_norm2), solver_.LogString(), e_str, e, nerr, derr, grad_norm2);
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
          goto closure;
        }
        if (options_.max_total_failures > 0 && out.num_failures >= options_.max_total_failures) {
          out.stop_reason = StopReason::kMaxFails;
          goto closure;
        }
        solver_.Failed();
      }

      // Detect if we need to stop and the reason
      if (solver_failed) {
        out.stop_reason = StopReason::kSolverFailed;
        goto closure;
      } else if (options_.min_error > 0 && err < options_.min_error) {
        out.stop_reason = StopReason::kMinError;
        goto closure;
      } else if (options_.min_delta_norm2 > 0 && dX_norm2 < options_.min_delta_norm2) {
        out.stop_reason = StopReason::kMinDeltaNorm;
        goto closure;
      } else if (options_.min_grad_norm2 > 0 && grad_norm2 < options_.min_grad_norm2) {
        out.stop_reason = StopReason::kMinGradNorm;
        goto closure;
      } else if (options_.stop_callback && options_.stop_callback(err, dX_norm2, grad_norm2)) {
        out.stop_reason = StopReason::kUserStopped;
        goto closure;
      }
    }
  closure:  // see mom? I'm using a goto ---->[]
    out.num_iters++;
    // Check for a time out
    out.duration_ms += toc_ms(t);
    if (options_.max_duration_ms > 0 && out.duration_ms > options_.max_duration_ms) {
      out.stop_reason = StopReason::kTimedOut;
    }
    // Return nullopt or the stop reason
    if (out.stop_reason == StopReason::kNone)
      return std::nullopt;
    else
      return out.stop_reason;
  }

 protected:
  /// Optimization options
  const Options options_;
  /// Linear solver
  SolverType solver_;
};

}  // namespace tinyopt::optimizers
