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

#include <ceres/jet.h>

namespace tinyopt {

/// The Automatix differentiation Jet struct
template <typename T, int N>
using Jet = ceres::Jet<T, N>;

// Traits

namespace traits {

// Check whether a type 'T' or '&T' is a Jet type
template <typename T>
struct is_jet_type {
  static constexpr bool value = false;
};

template <typename T, int N>
struct is_jet_type<Jet<T, N>> {
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
struct jet_details<Jet<T, N>> {
  using Scalar = T;
  static constexpr int Dims = N;
};

}  // namespace traits

/// Statically cast a type `T` to another one `T2`.
/// If `T2` is a Jet, we make sure the Jet.v is allocated to the total number of dimensions
template <typename T2, typename T>
inline auto StaticCast(const T &v, int total_dims) {
  using namespace traits;

  if constexpr (std::is_base_of_v<Eigen::MatrixBase<T>, T>) {
    auto o = v.template cast<T2>().eval();
    // Ceres'Jet does not support dynamic out of the box,so here we're setting jet.v size and
    // initialize with zeros
    if constexpr (is_jet_type_v<T2> && jet_details<T2>::Dims == Eigen::Dynamic) {
      for (int c = 0; c < v.cols(); ++c) {
        for (int r = 0; r < v.rows(); ++r) {
          o(r, c).v = Eigen::Vector<typename jet_details<T2>::Scalar, Eigen::Dynamic>::Zero(total_dims);
        }
      }
    }
    return o;
  } else {
    T2 o = static_cast<T2>(v);
    // Ceres'Jet does not support dynamic out of the box,so here we're setting jet.v size and
    // initialize with zeros
    if constexpr (is_jet_type_v<T2> && jet_details<T2>::Dims == Eigen::Dynamic) {
      o.v = Eigen::Vector<typename jet_details<T2>::Scalar, Eigen::Dynamic>::Zero(total_dims);
    }
    return o;
  }
}

/// Dynamically cast a type `T` to another one `T2`.
/// The method is disabled for Jet with dynamic size as the dimensions are missing.
template <typename T2, typename T,
          typename = std::enable_if_t<!tinyopt::traits::is_jet_type_v<T2> ||
                                      tinyopt::traits::jet_details<T2>::Dims != Eigen::Dynamic>>
inline auto StaticCast(const T &v) {
  if constexpr (std::is_base_of_v<Eigen::MatrixBase<T>, T>) {
    return v.template cast<T2>().eval();
  } else {
    return static_cast<T2>(v);
  }
}

}  // namespace tinyopt
