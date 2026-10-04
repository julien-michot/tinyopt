// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <type_traits>
#include <vector>

#include <tinyopt/types.h>

namespace tinyopt::c_api_detail {

/// Dynamic-size parameter copy optimized by the C API.
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
      if (plus_eq != nullptr) {
        for (tinyopt::Index index = 0; index < dims(); ++index)
          delta_values[static_cast<std::size_t>(index)] = delta[index];
        plus_eq(values.data(), delta_values.data());
      } else {
        for (tinyopt::Index index = 0; index < dims(); ++index) values[index] += delta[index];
      }
    } else {
      for (tinyopt::Index i = 0; i < dims(); ++i) values[i] += delta[i];
    }
    return *this;
  }
};

}  // namespace tinyopt::c_api_detail
