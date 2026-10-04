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
#include "c_api_params.h"
#include "c_api_problem.h"

namespace {

using tinyopt::c_api_detail::ParameterValues;

template <typename Scalar, typename Params, typename Problem>
tinyopt_status_t OptimizeDynamic(const Params *params, const Problem *problem,
                                 const tinyopt_options_t *c_options, tinyopt_summary_t *summary) {
  if (params == nullptr || problem == nullptr || params->x == nullptr || params->dims <= 0)
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
          result.num_diff_used || problem->type == TINYOPT_EVAL_COST_ONLY ? 1 : 0;
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

extern "C" TINYOPT_C_API tinyopt_status_t tinyopt_optimize(const tinyopt_params_t *params,
                                                           const tinyopt_problem_t *problem,
                                                           const tinyopt_options_t *options,
                                                           tinyopt_summary_t *summary) {
  return OptimizeDynamic<double>(params, problem, options, summary);
}

#if TINYOPT_C_API_ENABLE_FLOAT
extern "C" TINYOPT_C_API tinyopt_status_t tinyopt_optimizef(const tinyopt_paramsf_t *params,
                                                            const tinyopt_problemf_t *problem,
                                                            const tinyopt_options_t *options,
                                                            tinyopt_summary_t *summary) {
  return OptimizeDynamic<float>(params, problem, options, summary);
}
#endif

extern "C" TINYOPT_C_API tinyopt_status_t tinyopt_options_default(tinyopt_options_t *options) {
  if (options == nullptr) return TINYOPT_STATUS_INVALID_ARGUMENT;
  *options = tinyopt::c_api_detail::ToCOptions(tinyopt::Options{});
  options->log_error_symbol = "ε²";
  return TINYOPT_STATUS_OK;
}