// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <tinyopt/c/c_api_double.h>
#include <tinyopt/c/c_api_float.h>

#include <algorithm>
#include <exception>
#include <type_traits>
#include <vector>

#include <tinyopt/optimize.h>

#include "c_api_options.h"

namespace {

class ResidualCallbackError : public std::exception {};

template <typename ScalarType>
struct ParameterValues {
  using Scalar = ScalarType;
  static constexpr tinyopt::Index Dims = tinyopt::Dynamic;

  std::vector<Scalar> values;
  void (*plus_eq)(Scalar *, Scalar *) = nullptr;

  tinyopt::Index dims() const { return static_cast<tinyopt::Index>(values.size()); }

  template <typename TargetScalar>
  auto cast() const {
    ParameterValues<TargetScalar> result;
    result.values.reserve(values.size());
    for (const auto &value : values) result.values.emplace_back(static_cast<TargetScalar>(value));
    if constexpr (std::is_same_v<Scalar, TargetScalar>) result.plus_eq = plus_eq;
    return result;
  }

  ParameterValues &operator+=(const auto &delta) {
    if constexpr (std::is_floating_point_v<Scalar>) {
      tinyopt::Vector<Scalar, tinyopt::Dynamic> evaluated_delta = delta;
      plus_eq(values.data(), evaluated_delta.data());
    } else {
      for (tinyopt::Index i = 0; i < dims(); ++i) values[i] += delta[i];
    }
    return *this;
  }
};

template <typename Scalar, typename Residuals>
struct ResidualFunction {
  Residuals descriptor;

  tinyopt::Vector<Scalar, tinyopt::Dynamic> operator()(
      const ParameterValues<Scalar> &params) const {
    tinyopt::Vector<Scalar, tinyopt::Dynamic> residuals(descriptor.dims);
    if (descriptor.evaluate(params.values.data(), static_cast<int>(params.dims()), residuals.data(),
                            descriptor.dims, descriptor.user_data) != 0)
      throw ResidualCallbackError{};
    return residuals;
  }
};

template <typename Scalar, typename Params, typename Residuals>
tinyopt_status OptimizeDynamic(const Params *params, const Residuals *residuals,
                               const tinyopt_options *c_options, tinyopt_summary *summary) {
  if (params == nullptr || residuals == nullptr || params->x == nullptr || params->dims <= 0 ||
      params->plus_eq == nullptr || residuals->dims <= 0 || residuals->evaluate == nullptr)
    return TINYOPT_STATUS_INVALID_ARGUMENT;

  ParameterValues<Scalar> values;
  values.values.assign(params->x, params->x + params->dims);
  values.plus_eq = params->plus_eq;

  try {
    tinyopt::Options options = tinyopt::c_api_detail::ToTinyoptOptions(c_options);
    const auto result =
        tinyopt::Optimize(values, ResidualFunction<Scalar, Residuals>{*residuals}, options);

    if (summary != nullptr) {
      summary->stop_reason = static_cast<int>(result.stop_reason);
      summary->num_iters = result.num_iters;
      summary->num_failures = result.num_failures;
      summary->num_residuals = residuals->dims;
      summary->final_cost = result.final_cost;
      summary->used_numerical_differentiation = 1;
    }

    if (!result.Succeeded()) return TINYOPT_STATUS_OPTIMIZATION_FAILED;
    std::copy(values.values.begin(), values.values.end(), params->x);
    return TINYOPT_STATUS_OK;
  } catch (const ResidualCallbackError &) {
    return TINYOPT_STATUS_RESIDUAL_CALLBACK_FAILED;
  } catch (const std::exception &) {
    return TINYOPT_STATUS_INTERNAL_ERROR;
  } catch (...) {
    return TINYOPT_STATUS_INTERNAL_ERROR;
  }
}

}  // namespace

extern "C" TINYOPT_C_API tinyopt_status tinyopt_optimize(const tinyopt_params *params,
                                                         const tinyopt_residuals *residuals,
                                                         const tinyopt_options *options,
                                                         tinyopt_summary *summary) {
  return OptimizeDynamic<double>(params, residuals, options, summary);
}

#if TINYOPT_C_API_ENABLE_FLOAT
extern "C" TINYOPT_C_API tinyopt_status tinyopt_optimizef(const tinyopt_paramsf *params,
                                                          const tinyopt_residualsf *residuals,
                                                          const tinyopt_options *options,
                                                          tinyopt_summary *summary) {
  return OptimizeDynamic<float>(params, residuals, options, summary);
}
#endif

extern "C" TINYOPT_C_API tinyopt_status tinyopt_options_default(tinyopt_options *options) {
  if (options == nullptr) return TINYOPT_STATUS_INVALID_ARGUMENT;
  *options = tinyopt::c_api_detail::ToCOptions(tinyopt::Options{});
  options->log_error_symbol = "ε²";
  return TINYOPT_STATUS_OK;
}