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

#include <Eigen/src/Core/ArrayBase.h>
#include <Eigen/src/Core/MatrixBase.h>

namespace tinyopt::traits {

// Trait to check if a type is an Eigen matrix
template <typename T>
struct is_eigen_matrix_or_array
    : std::disjunction<std::is_base_of<Eigen::MatrixBase<T>, T>,
                       std::is_base_of<Eigen::ArrayBase<T>, T>> {};

template <typename T>
constexpr int is_eigen_matrix_or_array_v = is_eigen_matrix_or_array<T>::value;

// Trait to get the size of parameters at compile time

template <typename T, typename = void> struct params_size {
  static constexpr int value = T::Dims;
};


// Trait to get the size of parameters at compile time of a scalar (1)
template <typename T> struct params_size<T, std::enable_if_t<std::is_scalar_v<T>>> {
  static constexpr int value = 1;
};

// Trait to get the size of parameters at compile time of an Eigen Matrix
template <typename T>
struct params_size<
    T, std::enable_if_t<std::is_base_of_v<Eigen::MatrixBase<T>, T>>> {
  static constexpr int value =
      T::RowsAtCompileTime; // Get rows from Eigen matrix
};

template <typename T> constexpr int params_size_v = params_size<T>::value;


// Trait to get the dynamic dimensions/size

template <typename T, typename = void> struct params_dyn_size {
  int dims(const T &) const { return T::dims();}
};

// Trait to get the dynamic dimensions/size of a scalar (1)
template <typename T> struct params_dyn_size<T, std::enable_if_t<std::is_scalar_v<T>>> {
  constexpr int dims(const T &) const { return 1;}
};

// Trait to get the dynamic dimensions/size of an Eigen Matrix
template <typename T> struct params_dyn_size<T, std::enable_if_t<std::is_base_of_v<Eigen::MatrixBase<T>, T>>> {
  int dims(const T &m) const { return m.size();}
};

template <typename T> constexpr int params_size2_v = params_dyn_size<T>::value;

// Trait to get the Scalar


template <typename T, typename = void> struct params_scalar {
  using type = typename T::Scalar;
};

template <typename T> struct params_scalar<T, std::enable_if_t<std::is_scalar_v<T>>> {
  using type = T;
};

template <typename T>
struct params_scalar<
    T, std::enable_if_t<std::is_base_of_v<Eigen::MatrixBase<T>, T>>> {
  using type = typename T::Scalar; // Get Scalar type from Eigen matrix
};

template <typename T> using params_scalar_t = typename params_scalar<T>::type;

} // namespace tinyopt::traits
