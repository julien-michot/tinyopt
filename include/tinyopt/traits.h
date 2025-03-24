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

#include <type_traits>

namespace tinyopt::traits {

// Trait to check if a type is an Eigen matrix
template <typename T>
struct is_eigen_matrix_or_array : std::disjunction<
                                     std::is_base_of<Eigen::MatrixBase<T>, T>,
                                     std::is_base_of<Eigen::ArrayBase<T>, T>
                                   > {};

template <typename T>
constexpr int is_eigen_matrix_or_array_v = is_eigen_matrix_or_array<T>::value;

// Trait to get the size of parameters

template <typename T, typename = void> struct params_size {
  static constexpr int value =
      1; // Default is 0, to tell the user to define the trait
};

// Trait for Eigen Matrix
template <typename T>
struct params_size<
    T, std::enable_if_t<std::is_base_of_v<Eigen::MatrixBase<T>, T>>> {
  static constexpr int value =
      T::RowsAtCompileTime; // Get rows from Eigen matrix
};

template <typename T> constexpr int params_size_v = params_size<T>::value;

// Trait to get the Scalar

template <typename T, typename = void> struct params_scalar {
  using type = double; // Default is double
};

template <typename T>
struct params_scalar<
    T, std::enable_if_t<std::is_base_of_v<Eigen::MatrixBase<T>, T>>> {
  using type = typename T::Scalar; // Get Scalar type from Eigen matrix
};

template <typename T> using params_scalar_t = typename params_scalar<T>::type;

} // namespace tinyopt::traits
