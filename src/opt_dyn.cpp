// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <tinyopt/c/c_api_double.h>
#include <tinyopt/c/c_api_float.h>
#include <tinyopt/stop_reasons.h>

#include <algorithm>
#include <exception>
#include <type_traits>
#include <vector>

#include <tinyopt/optimize.h>

#include "c_api_options.h"
#include "c_api_problem.h"

namespace {

template <typename ScalarType>
struct ParameterValues {
  using Scalar = ScalarType;
  static constexpr tinyopt::Index Dims = tinyopt::Dynamic;

  std::vector<Scalar> values;
  std::vector<Scalar> delta_values;
  void (*plus_eq)(Scalar *, Scalar *) = nullptr;

  tinyopt::Index dims() const { return static_cast<tinyopt::Index>(values.size()); }

  template <typename TargetScalar>
  auto cast() const {
    ParameterValues<TargetScalar> result;
    result.values.reserve(values.size());
    result.delta_values.resize(values.size());
    for (const auto &value : values) result.values.emplace_back(static_cast<TargetScalar>(value));
    if constexpr (std::is_same_v<Scalar, TargetScalar>) result.plus_eq = plus_eq;
    return result;
  }

  ParameterValues &operator+=(const auto &delta) {
    if constexpr (std::is_floating_point_v<Scalar>) {
      for (tinyopt::Index i = 0; i < dims(); ++i)
        delta_values[static_cast<std::size_t>(i)] = delta[i];
      plus_eq(values.data(), delta_values.data());
    } else {
      for (tinyopt::Index i = 0; i < dims(); ++i) values[i] += delta[i];
    }
    return *this;
  }
};

template <typename Scalar, typename Params, typename Problem>
tinyopt_status OptimizeDynamic(const Params *params, const Problem *problem,
                               const tinyopt_options *c_options, tinyopt_summary *summary) {
  if (params == nullptr || problem == nullptr || params->x == nullptr || params->dims <= 0 ||
      params->plus_eq == nullptr)
    return TINYOPT_STATUS_INVALID_ARGUMENT;

  ParameterValues<Scalar> values;
  values.values.assign(params->x, params->x + params->dims);
  values.delta_values.resize(static_cast<std::size_t>(params->dims));
  values.plus_eq = params->plus_eq;

  try {
    tinyopt::Options options = tinyopt::c_api_detail::ToTinyoptOptions(c_options);
    const auto result = tinyopt::c_api_detail::OptimizeProblem(values, *problem, options);

    if (summary != nullptr) {
      summary->stop_reason = static_cast<int>(result.stop_reason);
      summary->num_iters = result.num_iters;
      summary->num_failures = result.num_failures;
      summary->num_residuals = problem->type == TINYOPT_EVAL_RESIDUALS ? problem->num_residuals : 1;
      summary->final_cost = result.final_cost;
      summary->used_numerical_differentiation =
          result.num_diff_used || problem->type == TINYOPT_EVAL_COST_ONLY ||
                  (problem->type == TINYOPT_EVAL_RESIDUALS && problem->use_jacobian == 0)
              ? 1
              : 0;
      summary->used_numerical_differentiation =
          result.num_diff_used || problem->type == TINYOPT_EVAL_COST_ONLY ||
                  (problem->type == TINYOPT_EVAL_RESIDUALS && problem->use_jacobian == 0)
              ? 1
              : 0;
    }

    if (!result.Succeeded()) return TINYOPT_STATUS_OPTIMIZATION_FAILED;
    std::copy(values.values.begin(), values.values.end(), params->x);
    return TINYOPT_STATUS_OK;
  } catch (const tinyopt::c_api_detail::UserStopRequested &) {
    std::copy(values.values.begin(), values.values.end(), params->x);
    if (summary != nullptr) {
      summary->stop_reason = static_cast<int>(tinyopt::StopReason::kUserStopped);
      summary->num_residuals = problem->type == TINYOPT_EVAL_RESIDUALS ? problem->num_residuals : 1;
    }
    return TINYOPT_STATUS_USER_STOPPED;
  } catch (const std::invalid_argument &) {
    return TINYOPT_STATUS_INVALID_ARGUMENT;
  } catch (const std::exception &) {
    return TINYOPT_STATUS_INTERNAL_ERROR;
  } catch (...) {
    return TINYOPT_STATUS_INTERNAL_ERROR;
  }
}

}  // namespace

extern "C" TINYOPT_C_API tinyopt_status tinyopt_optimize(const tinyopt_params *params,
                                                         const tinyopt_problem *problem,
                                                         const tinyopt_options *options,
                                                         tinyopt_summary *summary) {
  return OptimizeDynamic<double>(params, problem, options, summary);
}

#if TINYOPT_C_API_ENABLE_FLOAT
extern "C" TINYOPT_C_API tinyopt_status tinyopt_optimizef(const tinyopt_paramsf *params,
                                                          const tinyopt_problemf *problem,
                                                          const tinyopt_options *options,
                                                          tinyopt_summary *summary) {
  return OptimizeDynamic<float>(params, problem, options, summary);
}
#endif

extern "C" TINYOPT_C_API tinyopt_status tinyopt_options_default(tinyopt_options *options) {
  if (options == nullptr) return TINYOPT_STATUS_INVALID_ARGUMENT;
  *options = tinyopt::c_api_detail::ToCOptions(tinyopt::Options{});
  options->log_error_symbol = "ε²";
  return TINYOPT_STATUS_OK;
}