// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <tuple>
#include <type_traits>
#include <variant>
#include "tinyopt/math.h"
#include "tinyopt/stop_reasons.h"

#include <tinyopt/log.h>
#include <tinyopt/summary.h>
#include <tinyopt/time.h>
#include <tinyopt/traits.h>

#include <tinyopt/optimizers/options.h>

#ifndef TINYOPT_DISABLE_AUTODIFF
#include <tinyopt/diff/optimize_autodiff.h>
#endif
#ifndef TINYOPT_DISABLE_NUMDIFF
#include <tinyopt/diff/num_diff.h>
#endif

namespace tinyopt {

inline Options WithSolverOption(const Options &options, Options::Solver solver,
                                const char *optimizer_name) {
  Options normalized = options;
  if (normalized.solver_type != solver) {
    TINYOPT_LOG("⚠️ {} optimizer received a different solver option; using its configured "
                "algorithm",
                optimizer_name);
    normalized.solver_type = solver;
  }
  return normalized;
}

namespace detail {

#if defined(TINYOPT_ENFORCE_NO_DYNAMIC_ALLOCATIONS)
class EigenMallocGuard {
 public:
  explicit EigenMallocGuard(bool active)
      : active_(active), previous_(Eigen::internal::is_malloc_allowed()) {
    if (active_) Eigen::internal::set_is_malloc_allowed(false);
  }

  ~EigenMallocGuard() {
    if (active_) Eigen::internal::set_is_malloc_allowed(previous_);
  }

 private:
  bool active_;
  bool previous_;
};
#endif

template <typename Func, typename... Params>
struct PackedResidualAdapter {
  const Func &func;

  template <typename Pack>
  decltype(auto) operator()(const Pack &pack) const {
    return std::apply(func, pack.values);
  }
};

template <typename Func, typename... Params>
struct PackedAccumulationAdapter {
  const Func &func;

  template <typename Pack, typename Gradient, typename Hessian>
    requires std::is_invocable_v<const Func &, const Params &..., Gradient &, Hessian &>
  decltype(auto) operator()(const Pack &pack, Gradient &gradient, Hessian &hessian) const {
    return std::apply(
        [&](const auto &...params) { return std::invoke(func, params..., gradient, hessian); },
        pack.values);
  }

  template <typename Pack, typename Gradient>
    requires std::is_invocable_v<const Func &, const Params &..., Gradient &>
  decltype(auto) operator()(const Pack &pack, Gradient &gradient) const {
    return std::apply([&](const auto &...params) { return std::invoke(func, params..., gradient); },
                      pack.values);
  }
};

template <typename Func, typename... Params>
auto make_packed_adapter(const Func &func) {
  if constexpr (std::is_invocable_v<const Func &, const Params &...>)
    return PackedResidualAdapter<Func, Params...>{func};
  else
    return PackedAccumulationAdapter<Func, Params...>{func};
}

}  // namespace detail

/***
 *  @brief Optimizer
 */
template <typename Derived, typename Scalar_, Index Dims_, bool FirstOrder_, bool IsNLLS_>
class OptimizerCore {
 public:
  using Scalar = Scalar_;
  static constexpr Index Dims = Dims_;
  using Options = tinyopt::Options;

 public:
  explicit OptimizerCore(const Options &_options = {}) : options_{PrepareOptions(_options)} {}

  template <typename Gradient>
  void Clamp(Gradient &gradient, Scalar limit) const {
    if (limit == Scalar(0)) return;
    if constexpr (std::is_scalar_v<Gradient>)
      gradient = std::clamp(gradient, -limit, limit);
    else
      gradient = gradient.cwiseMax(-limit).cwiseMin(limit);
  }

  /// Initialize solver with specific gradient and hessian
  template <bool FO = FirstOrder_, std::enable_if_t<!FO, int> = 0>
  void InitWith(const auto &g, const auto &h) {
    derived().InitWith(g, h);
  }

  /// Initialize solver with specific gradient
  template <bool FO = FirstOrder_, std::enable_if_t<FO, int> = 0>
  void InitWith(const auto &g) {
    derived().InitWith(g);
  }

  /// Reset the optimization and solver
  void reset() { derived().reset(); }

  template <typename X_t>
  std::variant<StopReason, bool> ResizeIfNeeded(X_t &x) {
    const Index dims = traits::DynDims(x);  // Dynamic size
    if (Dims == Dynamic && dims == 0) {
      TINYOPT_LOG(
          "❌ Error: Parameters dimensions cannot be 0 or Dynamic at "
          "execution time");
      return StopReason::kSkipped;
    } else if (dims < 0) {
      TINYOPT_LOG("❌ Error: Parameters dimensions is negative: {} ", dims);
      return StopReason::kSkipped;
    }

    // Resize the solver if needed TODO move?
    bool resized = false;
    try {
      resized = derived().resize(dims);
    } catch (const std::bad_alloc &) {
      if (options_.log.enable) {
        int num_hessians = 1;
        if (options_.hessian.save_last) num_hessians++;
        TINYOPT_LOG(
            "❌ Failed to allocate {} Hessian(s) of size {}x{}, "
            "mem:{}GB, maybe use a SparseMatrix?",
            num_hessians, dims, dims, 1e-9f * static_cast<float>(dims * dims * sizeof(Scalar)));
      }
      return StopReason::kOutOfMemory;
    } catch (const std::invalid_argument &e) {
      TINYOPT_LOG("❌ Error: Failed to resize the linear solver. {}", e.what());
      return StopReason::kSkipped;
    }
    return resized;
  }

  /**
   * @brief Performs optimization on the given parameters `x` using the provided
   * cost or accumulation function `cost_or_acc`.
   *
   * This function is the primary interface for optimization within the library.
   * It takes the variable to be optimized and a function (or functor) that
   * calculates the cost function, its gradient, and optionally its Hessian.
   * The optimization process attempts to find a value of `x` that minimizes the
   * objective function.
   *
   * @tparam X_t The type of the parameters `x`.  This can be a scalar, a
   * vector, or a more complex data structure (e.g., a matrix).  The type must
   * support the necessary arithmetic operations for the chosen optimization
   * algorithm.
   * @tparam CostOrAccFunc The type of the cost or accumulation function
   * `cost_or_acc`.  This can be a function pointer, a lambda expression, or a
   * functor (an object that overloads the `operator()`).
   *
   * @param x The variable to be optimized.  This is passed by reference, and
   * the function will modify `x` in place to store the optimized value.
   * @param cost_or_acc The cost or accumulation function.  This function
   * calculates the cost function and, optionally, its derivatives.  The
   * `cost_or_acc` function can have one of the following signatures:
   * - `ResidualsType(const X_t& x)`:  Only the cost function  is
   * calculated.  In this case, the library will use automatic differentiation
   * (if available and enabled) or numerical differentiation to estimate the
   * gradient.
   * - `ScalarCost(const X_t& x, GradientType grad)`:
   * An accumulation function where the user must fill the gradient and return
   * the final cost (scalar or scalar+number of residuals)
   * - `ScalarCost(const X_t& x, GradientType grad, HessianType H)`:
   * An accumulation function where the user must fill the gradient and Hessian
   * and return the final cost (scalar or scalar+number of residuals)
   *
   * The `GradientType` and `HessianType` will be deduced from the type of `x`.
   * They can also be nullptr_t.
   * * ResidualsType is a scalar or Vector/Matrix of residuals (for NLLS).
   * * ScalarCost is the total cost/error or a pair (cost, number of residuals).
   * @param max_iters The maximum number of iterations to perform.  If this is
   * negative (the default), the optimization algorithm will run until a
   * convergence criterion is met.  A positive value limits the number of
   * iterations, which can be useful for preventing infinite loops or for
   * controlling the execution time.
   *
   * @note If the `cost_or_acc` function only provides the cost function or
   * residuals function, the library will automatically compute the gradient
   * (and approx. Hessian) using either automatic differentiation (if available
   * and enabled) or numerical differentiation. Automatic differentiation is
   * generally more accurate and efficient, but numerical differentiation can be
   * used as a fallback or when automatic differentiation is not supported.
   */
  template <typename T, typename U, typename... Rest, typename Func>
    requires(!std::is_same_v<std::remove_cvref_t<Func>, Options>)
  Summary Optimize(T &x, U &y, Rest &...rest, const Func &cost_or_acc) {
    tinyopt::ParamsPack pack(x, y, rest...);
    auto wrapped = detail::make_packed_adapter<Func, T, U, Rest...>(cost_or_acc);
    return OptimizeSingle(pack, wrapped);
  }

 private:
  template <typename X_t, typename CostOrAccFunc>
  Summary OptimizeSingle(X_t &x, const CostOrAccFunc &cost_or_acc, int max_iters = -1) {
#if defined(TINYOPT_ENFORCE_NO_DYNAMIC_ALLOCATIONS)
    detail::EigenMallocGuard eigen_malloc_guard(Dims != Dynamic);
#endif
    // Detect if we need to do  differentiation
    if constexpr (std::is_invocable_v<CostOrAccFunc, const X_t &>) {
      // Try to run AD
#ifndef TINYOPT_DISABLE_AUTODIFF
      using Jet = diff::Jet<Scalar, Dims>;
      using XJetType =
          std::conditional_t<std::is_floating_point_v<X_t>, Jet,
                             decltype(traits::params_trait<X_t>::template cast<Jet>(x))>;
      if constexpr (std::is_invocable_v<CostOrAccFunc, const XJetType &>) {
        using ResType = std::invoke_result_t<CostOrAccFunc, const XJetType &>;
        if constexpr (traits::is_tuple_v<ResType>) {
          const auto optimize = [&](auto &x, const auto &func, const auto &) {
            return OptimizeAcc(x, func, max_iters);
          };
          constexpr bool kIsNLLS = IsNLLS_;
          return tinyopt::OptimizeWithAutoDiff<kIsNLLS>(x, cost_or_acc, optimize, options_);
        } else if constexpr (traits::is_jet_type_v<ResType>) {
          const auto optimize = [&](auto &x, const auto &func, const auto &) {
            return OptimizeAcc(x, func, max_iters);
          };
          constexpr bool kIsNLLS = IsNLLS_;
          return tinyopt::OptimizeWithAutoDiff<kIsNLLS>(x, cost_or_acc, optimize, options_);
        } else if constexpr (traits::is_matrix_or_array_v<ResType>) {
          static_assert(traits::is_jet_type_v<typename ResType::Scalar>,
                        "Matrix residuals in autodiff must be Jet-valued");
          const auto optimize = [&](auto &x, const auto &func, const auto &) {
            return OptimizeAcc(x, func, max_iters);
          };
          constexpr bool kIsNLLS = IsNLLS_;
          return tinyopt::OptimizeWithAutoDiff<kIsNLLS>(x, cost_or_acc, optimize, options_);
        } else {
          Summary sum;
          sum.num_diff_used = true;
          if constexpr (FirstOrder_) {
            auto loss = diff::CreateNumDiffFunc1(x, cost_or_acc);
            sum = OptimizeAcc(x, loss, max_iters);
          } else {
            auto loss = diff::CreateNumDiffFunc2(x, cost_or_acc);
            sum = OptimizeAcc(x, loss, max_iters);
          }
          return sum;
        }
      } else {
        Summary sum;
        sum.num_diff_used = true;
        if constexpr (FirstOrder_) {
          auto loss = diff::CreateNumDiffFunc1(x, cost_or_acc);
          sum = OptimizeAcc(x, loss, max_iters);
        } else {
          auto loss = diff::CreateNumDiffFunc2(x, cost_or_acc);
          sum = OptimizeAcc(x, loss, max_iters);
        }
        return sum;
      }
#else
#ifndef TINYOPT_DISABLE_NUMDIFF
      Summary sum;
      sum.num_diff_used = true;
      if constexpr (FirstOrder_) {
        auto loss = diff::CreateNumDiffFunc1(x, cost_or_acc);
        sum = OptimizeAcc(x, loss, max_iters);
      } else {
        auto loss = diff::CreateNumDiffFunc2(x, cost_or_acc);
        sum = OptimizeAcc(x, loss, max_iters);
      }
      return sum;
#else
      throw std::invalid_argument(
          "Automatic and numerical differentiation are disabled for cost-only functions");
#endif
#endif  // TINYOPT_DISABLE_AUTODIFF
    } else {
      return OptimizeAcc(x, cost_or_acc, max_iters);
    }
  }

 public:
  template <typename X_t, typename CostOrAccFunc>
  Summary Optimize(X_t &x, const CostOrAccFunc &cost_or_acc, int max_iters = -1) {
    return OptimizeSingle(x, cost_or_acc, max_iters);
  }

  /**
   * @brief Performs optimization on the given parameters `x` using the provided
   * cost or accumulation function `cost_or_acc`.
   * See \ref Optimize for more informations.
   */
  template <typename T, typename U, typename... Rest, typename Func>
    requires(!std::is_same_v<std::remove_cvref_t<Func>, tinyopt::Options>)
  Summary operator()(T &x, U &y, Rest &...rest, const Func &cost_or_acc, int max_iters = -1) {
    tinyopt::ParamsPack pack(x, y, rest...);
    auto wrapped = detail::make_packed_adapter<Func, T, U, Rest...>(cost_or_acc);
    return OptimizeSingle(pack, wrapped, max_iters);
  }

  template <typename X_t, typename CostOrAccFunc>
  Summary operator()(X_t &x, const CostOrAccFunc &cost_or_acc, int max_iters = -1) {
    return OptimizeSingle(x, cost_or_acc, max_iters);
  }

  /**
   * @brief Performs optimization on the given parameters `x` using the provided
   * accumulation function `acc`.
   *
   * This function is the primary interface for optimization within the library
   * when the user manually updates the Gradient (and Hessian, if any). The
   * optimization process attempts to find a value of `x` that minimizes the
   * objective function.
   *
   * @tparam X_t The type of the parameters `x`.  This can be a scalar, a
   * vector, or a more complex data structure (e.g., a matrix).  The type must
   * support the necessary arithmetic operations for the chosen optimization
   * algorithm.
   * @tparam AccFunc The type of the accumulation function `acc`.  This can be
   * a function pointer, a lambda expression, or a functor (an object that
   * overloads the `operator()`).
   *
   * @param x The variable to be optimized.  This is passed by reference, and
   * the function will modify `x` in place to store the optimized value.
   * @param acc The accumulation function.  This function returns the total cost
   * must update the gradient(and Hessian, if any).  The `acc` function must
   * have one of the following signatures:
   * - `ScalarCost(const X_t& x, GradientType grad)`:
   * An accumulation function where the user must fill the gradient and return
   * the final cost (scalar or scalar+number of residuals)
   * - `ScalarCost(const X_t& x, GradientType grad, HessianType H)`:
   * An accumulation function where the user must fill the gradient and Hessian
   * and return the final cost (scalar or scalar+number of residuals)
   *
   * The `GradientType` and `HessianType` will be deduced from the type of `x`.
   * They can also be nullptr_t.
   * * ResidualsType is a scalar or Vector/Matrix of residuals (for NLLS).
   * * ScalarCost is the total cost/error or a pair (cost, number of residuals).
   * @param max_iters The maximum number of iterations to perform.  If this is
   * negative (the default), the optimization algorithm will run until a
   * convergence criterion is met.  A positive value limits the number of
   * iterations, which can be useful for preventing infinite loops or for
   * controlling the execution time.
   */
  template <typename X_t, typename AccFunc>
  Summary OptimizeAcc(X_t &x, const AccFunc &acc, int max_iters = -1) {
#if defined(TINYOPT_ENFORCE_NO_DYNAMIC_ALLOCATIONS)
    detail::EigenMallocGuard eigen_malloc_guard(Dims != Dynamic);
#endif
    using ptrait = traits::params_trait<X_t>;
    Summary sum;
    // Set start time
    sum.start_time = tic();
    if (max_iters < 0) max_iters = options_.stop.max_iters;
    max_iters++;  // +1 to potentially roll-back
    if (options_.opt.check_final_cost) max_iters++;

#if !defined(TINYOPT_ENFORCE_NO_DYNAMIC_ALLOCATIONS)
    sum.hist.errs.reserve(max_iters + 1);
    sum.hist.deltas2.reserve(max_iters + 1);
    sum.hist.successes.reserve(max_iters + 1);
#endif

    // Keep track of the last good 'x'
    constexpr bool kNoCopyX = true;  // TODO offer static alternative to the user
    using BestXType = std::conditional<kNoCopyX, std::nullptr_t, X_t>;
    BestXType *best_x = nullptr;
    if constexpr (!kNoCopyX) best_x = new X_t(x);  // using the copy constructor

    std::optional<Vector<Scalar, Dims>> last_dx;
    bool last_was_success = true;  // Last iteration was a success

    // Reports a parameter update to the user, who may ask to stop
    const auto notify_step = [&](const Vector<Scalar, Dims> &step, bool is_rollback) {
      if (options_.stop.step_callback &&
          options_.stop.step_callback(step.template cast<float>(), is_rollback) &&
          sum.stop_reason == StopReason::kNone)
        sum.stop_reason = StopReason::kUserStopped;
    };

    // Run several optimization iterations
    for (int iter = 0; iter < max_iters; ++iter) {
      const auto t = tic();
      const auto &[success, maybe_dx] = Step(x, acc, sum);
      bool eval_only = false;

      if (success) {  // Great, let's keep the good work

        ptrait::PlusEq(x, maybe_dx.value());  // Move X by dX
        notify_step(maybe_dx.value(), false);
        last_dx = maybe_dx.value();
        last_was_success = true;

        // On the very last iteration, we check that the final error is actually
        // lower
        if (options_.opt.check_final_cost && iter + 1 == max_iters) eval_only = true;

      } else {  // Failure to decrease error

        if (last_dx) {  // Roll-back
          if constexpr (kNoCopyX)
            ptrait::PlusEq(x, -last_dx.value());  // Move X by -dX
          else
            x = *best_x;
          notify_step(-last_dx.value(), true);
          last_dx.reset();
        } else if (maybe_dx) {                  // We failed several times in a row so just
                                                // evaluate the new x+dx
          ptrait::PlusEq(x, maybe_dx.value());  // Move X by dX
          notify_step(maybe_dx.value(), false);
          last_dx = maybe_dx.value();
        }

        eval_only = last_was_success == false;  // No need to build the linear system
        last_was_success = false;
      }

      derived().Rebuild(!eval_only);

      // Check for a time out
      sum.duration_ms += static_cast<float>(toc_ms(t));
      if (options_.stop.max_duration_ms > 0 && sum.duration_ms > options_.stop.max_duration_ms) {
        sum.stop_reason = StopReason::kTimedOut;
      }
      // Iteration done
      sum.num_iters++;
      // Stop now?
      if (sum.stop_reason != StopReason::kNone) break;
    }

    // Copy the very last hessian
    if constexpr (!FirstOrder_) {
#if defined(TINYOPT_ENFORCE_NO_DYNAMIC_ALLOCATIONS)
      if constexpr (Dims == Dynamic) {
        if (options_.hessian.save_last)
          sum.final_hessian = derived().Hessian().template cast<double>().eval();
      }
#else
      if (options_.hessian.save_last)
        sum.final_hessian = derived().Hessian().template cast<double>().eval();
#endif
    }

    if constexpr (!kNoCopyX) delete best_x;

    if (sum.stop_reason == StopReason::kNone && sum.num_iters >= max_iters)
      sum.stop_reason = StopReason::kMaxIters;
    // Print stop reason
    if (options_.log.enable && sum.stop_reason != StopReason::kNone)
      TINYOPT_LOG("{}, cost: [{}]", StopReasonDescription(sum, options_),
                  sum.final_cost.toString(options_.log.e, options_.log.print_inliers));
    return sum;
  }

  /// Run one optimization iteration, return the estimated next step (solve +
  /// decreased error)
  template <typename X_t, typename AccFunc>
  std::pair<bool, std::optional<Vector<Scalar, Dims>>> Step(X_t &x, const AccFunc &acc,
                                                            Summary &sum) {
    const auto iter = sum.num_iters;
    std::pair<bool, std::optional<Vector<Scalar, Dims>>> status{false, std::nullopt};

    // Set start time if not set already
    const auto t = tic();
    if (sum.start_time == TimePoint::min()) sum.start_time = t;

    // Resize the solver if needed
    const auto resize_status = ResizeIfNeeded(x);
    if (auto fail_reason = std::get_if<StopReason>(&resize_status)) {
      sum.stop_reason = *fail_reason;
      return status;
    }

    const bool resize_and_clear_solver = true;  // for now

    // Create the gradient and displacement `dx`
    Vector<Scalar, Dims> dx;
    Cost cost(NAN, sum.num_residuals);

    bool solver_failed = true;
    // Solver linear a few times until it's enough
    const uint8_t max_tries = options_.stop.max_consec_failures > 0
                                  ? std::max<uint8_t>(1, options_.stop.max_consec_failures)
                                  : 255;
    for (; sum.num_consec_failures <= max_tries;) {
      // Accumulate residuals and jacobians
      if (derived().Build(x, acc, resize_and_clear_solver)) {
        // Ok, let's try to solve for `dx` now
        if (const auto &maybe_dx = derived().Solve()) {
          dx = maybe_dx.value();  // TODO void copy?
          solver_failed = false;
        }
      }
      // Saves errors
      cost = derived().cost();
      // Check success/failure
      if (solver_failed) {  // Failure
        sum.num_consec_failures++;
        sum.num_failures++;
        // Check there's some residuals
        if (cost.num_resisuals == 0) {
          if (options_.log.enable) TINYOPT_LOG("❌ #{}: No residuals, stopping", iter);
          sum.stop_reason = StopReason::kSkipped;
          return status;
        } else if (std::isnan(cost.cost) || std::isinf(cost.cost)) {  // Check for NaNs and Inf
          if (options_.log.enable) TINYOPT_LOG("❌ #{}: NaN/Inf in error", iter);
          sum.stop_reason = StopReason::kSystemHasNaNOrInf;
          return status;
        } else if (options_.stop.max_consec_failures > 0 &&
                   sum.num_consec_failures >= options_.stop.max_consec_failures) {
          if (sum.final_cost < std::numeric_limits<Scalar>::max())
            sum.stop_reason = StopReason::kMaxConsecNoDecr;
          break;
        } else if (options_.log.enable)
          TINYOPT_LOG("❌ #{}:Failed to solve the linear system", iter);
        derived().FailedStep();
      } else {
        break;  // success -> we got a step!
      }
    }

    // Stop here if the solver failed constantly
    if (solver_failed) {
      sum.stop_reason = StopReason::kSolverFailed;
      return status;
    }

    const auto &err = cost.cost;
    const auto &nerr = cost.num_resisuals;

    // Check for NaNs and Inf
    if (std::isnan(err) || std::isinf(err)) {
      if (options_.log.enable) TINYOPT_LOG("❌ #{}: NaN/Inf in error: ε:{}", iter, err);
      sum.stop_reason = StopReason::kSystemHasNaNOrInf;
      return status;
    }

    // Check the displacement magnitude
    const double dx_norm2 = solver_failed ? 0 : dx.squaredNorm();
    const bool has_grad_norm2 = options_.stop.min_grad_norm2 > 0.0f ||
                                options_.stop.stop_callback || options_.stop.stop_callback2;
    const double grad_norm2 = has_grad_norm2 ? derived().GradientSquaredNorm() : 0.0;
    if (std::isnan(dx_norm2) || std::isinf(dx_norm2)) {
      if (options_.log.enable && options_.log.print_failure) {
        TINYOPT_LOG("❌ Failure, dX = \n{}", dx.template cast<float>());
        TINYOPT_LOG("Solver: {}", derived().stateAsString());
        TINYOPT_LOG("grad = \n{}", derived().Gradient());
        if constexpr (!FirstOrder_) TINYOPT_LOG("H = \n{}", derived().H());
      }
      sum.stop_reason = StopReason::kSystemHasNaNOrInf;
      return status;
    }

    // Cost change (negative is good)
    const double derr = err - sum.final_cost;
    const bool is_good_step = derr < Scalar(0.0);
    // Relative Cost change, defined as (εp-ε)/εp, εp is previous cost,
    const double rel_derr = sum.final_cost > FloatEpsilon<Scalar>() &&
                                    sum.final_cost < std::numeric_limits<Scalar>::max()
                                ? (sum.final_cost - err) / sum.final_cost
                                : 0.0f;
    // Save history of errors and deltas
    sum.hist.Add(err, dx_norm2, is_good_step);

    // Update output struct
    if (is_good_step || iter == 0) { /* GOOD Step */
      // Note: we guess it's a good step in the first iteration
      if (iter > 0) derived().GoodStep(options_.opt.use_step_quality_approx ? rel_derr : 0.0f);
      sum.num_consec_failures = 0;
      sum.final_cost = cost;
      sum.final_rerr_dec = rel_derr;
    } else { /* BAD Step */
      derived().BadStep();
      sum.num_failures++;
      sum.num_consec_failures++;
      if (options_.stop.max_consec_failures > 0 &&
          sum.num_consec_failures >= options_.stop.max_consec_failures) {
        sum.stop_reason = StopReason::kMaxConsecNoDecr;
      }
      if (options_.stop.max_total_failures > 0 &&
          sum.num_failures >= options_.stop.max_total_failures) {
        sum.stop_reason = StopReason::kMaxNoDecr;
        return status;
      }
    }

    // Log
    if (options_.log.enable) {
      std::ostringstream oss;
      if (options_.log.print_emoji) oss << (is_good_step ? (iter == 0 ? "ℹ️" : "✅") : "❌");
      oss << "#" << iter << " ";
      if (options_.log.print_x) {
        if constexpr (traits::is_scalar_v<X_t>) {
          oss << TINYOPT_FORMAT_NS::format("x:{:.5f} ", x);
        } else if constexpr (traits::is_matrix_or_array_v<X_t>) {  // Flattened X
          oss << "x:["
#ifdef TINYOPT_NO_FORMATTERS
              << x.reshaped().transpose()
#else
              << TINYOPT_FORMAT_NS::format("{}", x.reshaped().transpose())
#endif  // TINYOPT_NO_FORMATTERS
              << "] ";
        } else if constexpr (traits::is_streamable_v<X_t>) {
          // User must define the stream operator of ParameterType
          oss << "{" << x << "} ";
        }
      } else if constexpr (Dims == Dynamic) {
        const auto dims = traits::DynDims(x);
        oss << "x:ℝ^" << dims << " ";
        if (derived().dims() != dims) oss << "∇:ℝ^" << dims << " ";
      }

      // Print error/cost
      oss << TINYOPT_FORMAT_NS::format("{}:{:.4e} n:{} d{}:{:+.2e} r{}:{:+.1e} ", options_.log.e,
                                       err, nerr, options_.log.e, iter == 0 ? 0.0f : derr,
                                       options_.log.e, rel_derr);

      // Print step info
      oss << TINYOPT_FORMAT_NS::format("|δx|:{:.2e} ", sqrt(dx_norm2));
      if (options_.log.print_dx) oss << TINYOPT_FORMAT_NS::format("δx:[{}] ", dx);
      // Estimate max standard deviations from (co)variances
      if constexpr (!FirstOrder_) {
        if (is_good_step && options_.log.print_max_stdev)
          oss << TINYOPT_FORMAT_NS::format("⎡σ⎤:{:.2f} ", derived().MaxStdDev());
      }
      // Print gradient
      if (has_grad_norm2) oss << TINYOPT_FORMAT_NS::format("|∇|:{:.2e} ", sqrt(grad_norm2));
      // Print Solver state
      oss << derived().stateAsString();
      // Print inliers
      if (options_.log.print_inliers) {
        oss << TINYOPT_FORMAT_NS::format("in:{:.2f}% ({}) ", cost.inlier_ratio * 100.0,
                                         cost.NumInliers());
      }
      // Print extra log
      if (!cost.log_str.empty()) oss << cost.log_str << " ";
      // Print timing
      if (options_.log.print_t) oss << TINYOPT_FORMAT_NS::format("τ:{:.2f} ", sum.duration_ms);
      // Print now!
      TINYOPT_LOG("{}", oss.str());
    }

    // Detect if we need to stop
    if (sum.stop_reason == StopReason::kNone) {
      if (solver_failed)
        sum.stop_reason = StopReason::kSolverFailed;
      else if (options_.stop.min_error > 0 && err < options_.stop.min_error)
        sum.stop_reason = StopReason::kMinError;
      else if (options_.stop.min_rerr_dec > 0 && rel_derr > 0.0 &&
               rel_derr < options_.stop.min_rerr_dec)
        sum.stop_reason = StopReason::kMinRelError;
      else if (options_.stop.min_step_norm2 > 0 && dx_norm2 < options_.stop.min_step_norm2)
        sum.stop_reason = StopReason::kMinDeltaNorm;
      else if (options_.stop.min_grad_norm2 > 0 && grad_norm2 < options_.stop.min_grad_norm2)
        sum.stop_reason = StopReason::kMinGradNorm;
      else if (options_.stop.stop_callback &&
               options_.stop.stop_callback(err, dx_norm2, grad_norm2))
        sum.stop_reason = StopReason::kUserStopped;
      else if (options_.stop.stop_callback2 &&
               options_.stop.stop_callback2(float(err), dx.template cast<float>(),
                                            derived().Gradient().template cast<float>()))
        sum.stop_reason = StopReason::kUserStopped;
    }

    status.first = is_good_step;
    status.second = dx;
    return status;
  }

  Derived &optimizer() { return derived(); }
  const Derived &optimizer() const { return derived(); }

 protected:
  static Options PrepareOptions(const Options &options) {
    Options prepared = options;
#if defined(TINYOPT_ENFORCE_NO_DYNAMIC_ALLOCATIONS)
    if constexpr (Dims != Dynamic) {
      prepared.log.enable = false;
      prepared.hessian.save_last = false;
    }
#endif
    return prepared;
  }

  /// Optimization options
  const Options options_;
 private:
  Derived &derived() { return static_cast<Derived &>(*this); }
  const Derived &derived() const { return static_cast<const Derived &>(*this); }
};

}  // namespace tinyopt
