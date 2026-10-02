// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>
#include <utility>

#include <tinyopt/traits/parameter_traits.h>

namespace tinyopt {

template <typename Params>
class ParamsWrapper {
 public:
  using WrappedParams = std::remove_cvref_t<Params>;
  using Scalar = typename traits::params_trait<WrappedParams>::Scalar;
  static constexpr Index Dims = Dynamic;

  explicit ParamsWrapper(Params params) : params(std::forward<Params>(params)) {}

  [[nodiscard]] Index dims() const {
    if constexpr (traits::params_trait<WrappedParams>::Dims == Dynamic)
      return traits::params_trait<WrappedParams>::dims(params);
    else
      return traits::params_trait<WrappedParams>::Dims;
  }

  template <typename T>
  [[nodiscard]] auto cast() const {
    auto casted = traits::params_trait<WrappedParams>::template cast<T>(params);
    using CastParams = std::decay_t<decltype(casted)>;
    return ParamsWrapper<CastParams>(std::move(casted));
  }

  ParamsWrapper& operator+=(const auto& delta) {
    params += delta;
    return *this;
  }

  Params params;
};

}  // namespace tinyopt