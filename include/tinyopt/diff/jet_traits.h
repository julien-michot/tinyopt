// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tinyopt/diff/jet.h>
#include <tinyopt/traits.h>

namespace tinyopt::traits {

template <typename T, int N>
struct is_scalar<diff::Jet<T, N>> {
  static constexpr bool value = true;
};

// Check whether a type 'T' or '&T' is a Jet type
template <typename T>
struct is_jet_type {
  static constexpr bool value = false;
};

template <typename T, int N>
struct is_jet_type<diff::Jet<T, N>> {
  static constexpr bool value = true;
};

template <typename T>
inline constexpr bool is_jet_type_v = is_jet_type<T>::value;

// Check whether a type 'T' or '&T' is a Jet type
template <typename T>
struct jet_details {
  using Scalar = double;
  static constexpr Index Dims = 0;
};

template <typename T, int N>
struct jet_details<diff::Jet<T, N>> {
  using Scalar = T;
  static constexpr Index Dims = N;
};

// Trait specialization for Jet
template <typename _Scalar, int N>
struct params_trait<diff::Jet<_Scalar, N>> {
  using T = diff::Jet<_Scalar, N>;
  using Scalar = _Scalar;  // The scalar type
  static constexpr Index Dims =
      params_trait<Scalar>::Dims == Dynamic ? Dynamic : 1;  // Compile-time parameters dimensions
  // Execution-time parameters dimensions
  static auto dims(const T& v) { return params_trait<Scalar>::dims(v); }
  // Cast to a new type, only needed when using automatic differentiation
  template <typename T2>
  static auto cast(const T& v) {
    if constexpr (has_static_cast_v<T>)
      return T::template cast<T2>(v);
    else if constexpr (has_cast_v<T>)
      return v.template cast<T2>();
    else
      return T2(v);  // use casting operator  by default
  }
  // Define update / manifold
  static void PlusEq(T& v, const auto& delta) { v += delta; }
};

}  // namespace tinyopt::traits
