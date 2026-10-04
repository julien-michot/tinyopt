// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <exception>
#include <type_traits>

#include <tinyopt/optimize.h>

#include "c_api_options.h"
#include "c_api_problem.h"

namespace tinyopt::c_api_detail {

class ResidualCallbackError : public std::exception {};

template <typename ScalarType, int Dimension>
struct FixedParameterValues {
  using Scalar = ScalarType;
  static constexpr Index Dims = Dimension;

  std::array<Scalar, Dimension> values{};
  void (*plus_eq)(Scalar *, Scalar *) = nullptr;

  Index dims() const { return Dimension; }

  template <typename TargetScalar>
  auto cast() const {
    FixedParameterValues<TargetScalar, Dimension> result;
    for (int i = 0; i < Dimension; ++i)
      result.values[static_cast<std::size_t>(i)] = static_cast<TargetScalar>(values[i]);
    if constexpr (std::is_same_v<Scalar, TargetScalar>) result.plus_eq = plus_eq;
    return result;
  }

  FixedParameterValues &operator+=(const auto &delta) {
    if constexpr (std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double>) {
      Vector<Scalar, Dimension> step = delta;
      plus_eq(values.data(), step.data());
    } else {
      for (int i = 0; i < Dimension; ++i) values[static_cast<std::size_t>(i)] += delta[i];
    }
    return *this;
  }
};

template <typename Scalar, int Dimension, typename Problem>
tinyopt_status OptimizeFixed(Scalar *x, void (*plus_eq)(Scalar *, Scalar *), const Problem *problem,
                             const tinyopt_options *c_options, tinyopt_summary *summary) {
  if (x == nullptr || plus_eq == nullptr || problem == nullptr)
    return TINYOPT_STATUS_INVALID_ARGUMENT;

  FixedParameterValues<Scalar, Dimension> params;
  params.plus_eq = plus_eq;
  for (int i = 0; i < Dimension; ++i) params.values[static_cast<std::size_t>(i)] = x[i];

  try {
    Options options = ToTinyoptOptions(c_options);
    const auto result = OptimizeProblem(params, *problem, options);

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
    }
    if (!result.Succeeded()) return TINYOPT_STATUS_OPTIMIZATION_FAILED;

    for (int i = 0; i < Dimension; ++i) x[i] = params.values[static_cast<std::size_t>(i)];
    return TINYOPT_STATUS_OK;
  } catch (const UserStopRequested &) {
    for (int i = 0; i < Dimension; ++i) x[i] = params.values[static_cast<std::size_t>(i)];
    if (summary != nullptr) {
      summary->stop_reason = static_cast<int>(StopReason::kUserStopped);
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

}  // namespace tinyopt::c_api_detail