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

#include <tinyopt/math.h>

namespace tinyopt::traits {

// Check whether a type 'T' or '&T' is nullptr_t
template <typename T>
struct is_nullptr_type : std::is_same<std::remove_reference_t<T>, std::nullptr_t> {};
template <typename T>
inline constexpr bool is_nullptr_type_v = is_nullptr_type<T>::value;

// Trait to detect std::pair
template <typename T>
struct is_pair : std::false_type {};
template <typename T, typename U>
struct is_pair<std::pair<T, U>> : std::true_type {};
template <typename T>
inline constexpr bool is_pair_v = is_pair<T>::value;

// Trait to check if a type is an Eigen matrix
template <typename T>
struct is_matrix_or_array
    : std::disjunction<std::is_base_of<MatrixBase<T>, T>, std::is_base_of<ArrayBase<T>, T>> {};

template <typename T>
constexpr bool is_matrix_or_array_v = is_matrix_or_array<T>::value;

// Logging trait

template <typename T, typename = void>
struct is_streamable : std::false_type {};

template <typename T>
struct is_streamable<
    T,
    typename std::enable_if<std::is_convertible<
        decltype(std::declval<std::ostream&>() << std::declval<T>()), std::ostream&>::value>::type>
    : std::true_type {};

template <typename T>
constexpr bool is_streamable_v = is_streamable<T>::value;

// Default parameters trait

template <typename T, typename = void>
struct params_trait {
  using Scalar = typename T::Scalar;    // The scalar type
  static constexpr int Dims = T::Dims;  // Compile-time parameters dimensions

  // Execution-time parameters dimensions
  static int dims(const T& v) { return Dims == Dynamic ? v.dims() : Dims; }

  // Cast to a new type, only needed when using automatic differentiation
  template <typename T2>
  static auto cast(const T& v) {
    return v.template cast<T2>();
  }

  // Define update / manifold
  static void pluseq(T& v, const Vector<Scalar, Dims>& delta) { v += delta; }
};

// Trait specialization for scalar (float, double)
template <typename T>
struct params_trait<T, std::enable_if_t<std::is_scalar_v<T>>> {
  using Scalar = T;               // The scalar type
  static constexpr int Dims = 1;  // Compile-time parameters dimensions
  // Execution-time parameters dimensions
  static constexpr int dims(const T&) { return 1; }
  // Cast to a new type, only needed when using automatic differentiation
  template <typename T2>
  static T2 cast(const T& v) {
    return T2(v);
  }
  // Define update / manifold
  static void pluseq(T& v, const Vector<Scalar, Dims>& delta) { v += delta[0]; }
  static void pluseq(T& v, const Scalar& delta) { v += delta; }
};

// Trait specialization for MatrixBase
template <typename T>
struct params_trait<T, std::enable_if_t<is_matrix_or_array_v<T>>> {
  using Scalar = typename T::Scalar;  // The scalar type
  static constexpr int Dims =
      (T::RowsAtCompileTime == Dynamic || T::ColsAtCompileTime == Dynamic)
          ? Dynamic
          : T::RowsAtCompileTime * T::ColsAtCompileTime;  // Compile-time parameters dimensions
  // Execution-time parameters dimensions
  static int dims(const T& m) { return m.size(); }

  // Cast to a new type, only needed when using automatic differentiation
  template <typename T2>
  static auto cast(const T& v) {
    return v.template cast<T2>().eval();
  }
  // Define update / manifold
  static void pluseq(T& v, const Vector<Scalar, Dims>& delta) {
    if constexpr (Dims == Dynamic) assert(delta.rows() == (int)v.size());
    if constexpr (T::ColsAtCompileTime == 1)
      v += delta;
    else
      v += delta.reshaped(v.rows(), v.cols());
  }
};

// Trait specialization for std vector
template <typename _Scalar>
struct params_trait<std::vector<_Scalar>> {
  using T = typename std::vector<_Scalar>;
  using Scalar = _Scalar;               // The scalar type
  static constexpr int Dims = Dynamic;  // Compile-time parameters dimensions
  // Execution-time parameters dimensions
  static int dims(const T& v) { return v.size(); }
  // Cast to a new type, only needed when using automatic differentiation
  template <typename T2>
  static auto cast(const T& v) {
    std::vector<T2> o(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) o[i] = static_cast<T2>(v[i]);
    return o;
  }
  // Define update / manifold
  static void pluseq(T& v, const Vector<Scalar, Dims>& delta) {
    if constexpr (Dims == Dynamic) assert(delta.rows() == (int)v.size());
    for (std::size_t i = 0; i < v.size(); ++i) v[i] += delta[i];
  }
};

// Trait specialization for std::array
template <typename _Scalar, std::size_t N>
struct params_trait<std::array<_Scalar, N>> {
  using T = typename std::array<_Scalar, N>;
  using Scalar = _Scalar;         // The scalar type
  static constexpr int Dims = N;  // Compile-time parameters dimensions
  // Cast to a new type, only needed when using automatic differentiation
  template <typename T2>
  static auto cast(const T& v) {
    std::array<T2, N> o;
    for (std::size_t i = 0; i < N; ++i) o[i] = static_cast<T2>(v[i]);
    return o;
  }
  // Define update / manifold
  static void pluseq(T& v, const Vector<Scalar, Dims>& delta) {
    for (std::size_t i = 0; i < N; ++i) v[i] += delta[i];
  }
};

}  // namespace tinyopt::traits
