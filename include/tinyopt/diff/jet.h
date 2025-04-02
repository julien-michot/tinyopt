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

#include <tinyopt/3rdparty/ceres/jet.h>  // should not be another one
#include "tinyopt/math.h"

namespace tinyopt::diff {

/// The Automatix differentiation Jet struct
template <typename T, int N>
using Jet = ceres::Jet<T, N>;

}  // namespace tinyopt::diff
// Traits

namespace tinyopt::traits {

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
  static constexpr int Dims = 0;
};

template <typename T, int N>
struct jet_details<diff::Jet<T, N>> {
  using Scalar = T;
  static constexpr int Dims = N;
};

// Trait specialization for Jet
template <typename _Scalar, int N>
struct params_trait<diff::Jet<_Scalar, N>> {
  using T = diff::Jet<_Scalar, N>;
  using Scalar = _Scalar;  // The scalar type
  static constexpr int Dims =
      params_trait<Scalar>::Dims == Dynamic ? Dynamic : 1;  // Compile-time parameters dimensions
  // Execution-time parameters dimensions
  static auto dims(const T& v) { return params_trait<Scalar>::dims(v); }
  // Cast to a new type, only needed when using automatic differentiation
  template <typename T2>
  static auto cast(const T& v) {
    return T2(v);
  }
  // Define update / manifold
  static void pluseq(T& v, const auto& delta) { v += delta; }
};

}  // namespace tinyopt::traits
