// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <exception>
#include <type_traits>

#include <tinyopt/optimize.h>

#include "c_api_options.h"

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

template <typename Scalar, int Dimension, typename Callback>
struct FixedResidualFunction {
  Callback evaluate;
  int dims;
  void *user_data;

  Vector<Scalar, Dynamic> operator()(const FixedParameterValues<Scalar, Dimension> &params) const {
    Vector<Scalar, Dynamic> residuals(dims);
    if (evaluate(params.values.data(), static_cast<int>(params.dims()), residuals.data(), dims,
                 user_data) != 0)
      throw ResidualCallbackError{};
    return residuals;
  }
};

template <typename Scalar, int Dimension, typename Callback>
tinyopt_status OptimizeFixed(Scalar *x, void (*plus_eq)(Scalar *, Scalar *), int residual_dims,
                             Callback evaluate, void *user_data, const tinyopt_options *c_options,
                             tinyopt_summary *summary) {
  if (x == nullptr || plus_eq == nullptr || residual_dims <= 0 || evaluate == nullptr)
    return TINYOPT_STATUS_INVALID_ARGUMENT;

  FixedParameterValues<Scalar, Dimension> params;
  params.plus_eq = plus_eq;
  for (int i = 0; i < Dimension; ++i) params.values[static_cast<std::size_t>(i)] = x[i];

  try {
    Options options = ToTinyoptOptions(c_options);
    const auto result = Optimize(
        params,
        FixedResidualFunction<Scalar, Dimension, Callback>{evaluate, residual_dims, user_data},
        options);

    if (summary != nullptr) {
      summary->stop_reason = static_cast<int>(result.stop_reason);
      summary->num_iters = result.num_iters;
      summary->num_failures = result.num_failures;
      summary->num_residuals = residual_dims;
      summary->final_cost = result.final_cost;
      summary->used_numerical_differentiation = 1;
    }
    if (!result.Succeeded()) return TINYOPT_STATUS_OPTIMIZATION_FAILED;

    for (int i = 0; i < Dimension; ++i) x[i] = params.values[static_cast<std::size_t>(i)];
    return TINYOPT_STATUS_OK;
  } catch (const ResidualCallbackError &) {
    return TINYOPT_STATUS_RESIDUAL_CALLBACK_FAILED;
  } catch (const std::exception &) {
    return TINYOPT_STATUS_INTERNAL_ERROR;
  } catch (...) {
    return TINYOPT_STATUS_INTERNAL_ERROR;
  }
}

}  // namespace tinyopt::c_api_detail