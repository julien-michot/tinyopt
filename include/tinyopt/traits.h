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
struct is_nullptr_t : std::is_same<std::decay_t<T>, std::nullptr_t> {};
template <typename T>
inline constexpr bool is_nullptr_v = is_nullptr_t<T>::value;

// Check whether a type 'T' or '&T' is a bool
template <typename T>
struct is_bool : std::is_same<std::decay_t<T>, bool> {};
template <typename T>
inline constexpr bool is_bool_v = is_bool<T>::value;

// Check whether a type 'T' or '&T' is a bool
template <typename T>
struct is_scalar : std::is_scalar<std::decay_t<T>> {};
template <typename T>
inline constexpr bool is_scalar_v = is_scalar<T>::value;

// Trait to detect std::pair
template <typename T>
struct is_pair : std::false_type {};
template <typename T, typename U>
struct is_pair<std::pair<T, U>> : std::true_type {};
template <typename T>
inline constexpr bool is_pair_v = is_pair<std::decay_t<T>>::value;

// Trait to check if a type is a Matrix/Vector
template <typename T>
struct is_matrix_or_array
    : std::disjunction<std::is_base_of<MatrixBase<T>, T>, std::is_base_of<ArrayBase<T>, T>> {};

template <typename T>
constexpr bool is_matrix_or_array_v = is_matrix_or_array<std::decay_t<T>>::value;

// Trait to check if a type is a Sparse Matrix
template <typename T>
struct is_sparse_matrix : std::false_type {};
template <typename T>
struct is_sparse_matrix<SparseMatrix<T>> : std::true_type {};

template <typename T>
constexpr bool is_sparse_matrix_v = is_sparse_matrix<std::decay_t<T>>::value;

// Trait to check if a type is a Matrix/Vector
template <typename T>
constexpr bool is_matrix_or_scalar_v = (std::is_scalar_v<T> && !std::is_same_v<T, bool>) ||
                                       is_sparse_matrix_v<T> || is_matrix_or_array_v<T>;

// Logging trait

// NOTE this trait is not working for local struct...

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

// Check whether a type has the static cast method

template <typename T, typename = void>
struct has_cast : std::false_type {};

template <typename T>
struct has_cast<T, std::void_t<decltype(std::declval<const T>().template cast<int>())>>
    : std::true_type {};

// Helper variable template for easier usage
template <typename T>
inline constexpr bool has_cast_v = has_cast<T>::value;

template <typename T, typename = void>
struct has_static_cast : std::false_type {};

template <typename T>
struct has_static_cast<T, std::void_t<decltype(T::template cast<float>(std::declval<const T&>()))>>
    : std::true_type {};

// Helper variable template for easier usage
template <typename T>
inline constexpr bool has_static_cast_v = has_static_cast<T>::value;

// Parameters traits

template <typename T, typename = void>
struct params_trait {
  using Scalar = typename T::Scalar;      // The scalar type
  static constexpr Index Dims = Dynamic;  // Compile-time parameters dimensions

  // Execution-time parameters dimensions
  static Index dims(const T& v) { return v.dims(); }

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

template <typename T>
struct params_trait<T, std::void_t<decltype(T::Dims)>> {
  using Scalar = typename T::Scalar;      // The scalar type
  static constexpr Index Dims = T::Dims;  // Compile-time parameters dimensions

  // Execution-time parameters dimensions
  static Index dims(const T& v) { return Dims == Dynamic ? v.dims() : Dims; }

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

// Trait specialization for scalar (float, double)
template <typename T>
struct params_trait<T, std::enable_if_t<std::is_scalar_v<T>>> {
  using Scalar = T;                 // The scalar type
  static constexpr Index Dims = 1;  // Compile-time parameters dimensions
  // Execution-time parameters dimensions
  static constexpr Index dims(const T&) { return 1; }
  // Cast to a new type, only needed when using automatic differentiation
  template <typename T2>
  static T2 cast(const T& v) {
    return static_cast<T2>(v);
  }
  // Define update / manifold
  static void PlusEq(T& v, const auto& delta) { v += delta[0]; }
  static void PlusEq(T& v, const Scalar& delta) { v += delta; }
};

// Trait specialization for MatrixBase
template <typename T>
struct params_trait<T, std::enable_if_t<is_matrix_or_array_v<T>>> {
  using Scalar = typename T::Scalar;  // The scalar type
  static constexpr int ColsAtCompileTime = T::ColsAtCompileTime;
  static constexpr Index Dims =
      (T::RowsAtCompileTime == Dynamic || T::ColsAtCompileTime == Dynamic)
          ? Dynamic
          : T::RowsAtCompileTime * T::ColsAtCompileTime;  // Compile-time parameters dimensions
  // Execution-time parameters dimensions
  static Index dims(const T& m) { return m.size(); }

  // Cast to a new type, only needed when using automatic differentiation
  template <typename T2>
  static auto cast(const T& v) {
    return v.template cast<T2>().eval();
  }
  // Define update / manifold
  static void PlusEq(T& v, const auto& delta) {
    if constexpr (Dims == Dynamic) assert(delta.rows() == (int)v.size());
    if constexpr (T::ColsAtCompileTime == 1)
      v += delta;
    else
      v += delta.reshaped(v.rows(), v.cols());
  }
};

// Trait specialization for SparseMatrix
template <typename T>
struct params_trait<T, std::enable_if_t<is_sparse_matrix_v<T>>> {
  using Scalar = typename T::Scalar;      // The scalar type
  static constexpr Index Dims = Dynamic;  // Compile-time parameters dimensions

  // Execution-time parameters dimensions
  static Index dims(const T& m) { return m.size(); }

  // Cast to a new type, only needed when using automatic differentiation
  template <typename T2>
  static auto cast(const T& v) {
    return v.template cast<T2>().eval();
  }
  // Define update / manifold
  static void PlusEq(T& v, const auto& delta) {
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
  using Scalar = _Scalar;  // The scalar type
  using ScalarParamsTraits = params_trait<Scalar>;
  static constexpr Index Dims = Dynamic;  // Compile-time parameters dimensions
  // Execution-time parameters dimensions
  static Index dims(const T& v) {
    constexpr int ScalarDims = ScalarParamsTraits::Dims;
    if constexpr (std::is_scalar_v<Scalar> || ScalarDims == 1) {
      return static_cast<int>(v.size());
    } else if constexpr (ScalarDims == Dynamic) {
      int d = 0;
      for (std::size_t i = 0; i < v.size(); ++i) d += ScalarParamsTraits::dims(v[i]);
      return d;
    } else {
      return static_cast<int>(v.size()) * ScalarDims;
    }
  }
  // Cast to a new type, only needed when using automatic differentiation
  template <typename T2>
  static auto cast(const T& v) {
    using Scalar2 =
        std::decay_t<decltype(ScalarParamsTraits::template cast<T2>(std::declval<Scalar>()))>;
    std::vector<Scalar2> o;
    o.reserve(v.size());
    for (auto& x : v) o.emplace_back(ScalarParamsTraits::template cast<T2>(x));
    return o;
  }
  // Define update / manifold
  static void PlusEq(T& v, const auto& delta) {
    for (std::size_t i = 0; i < v.size(); ++i) {
      if constexpr (std::is_scalar_v<Scalar> || ScalarParamsTraits::Dims == 1)
        v[i] += delta[i];
      else if constexpr (ScalarParamsTraits::Dims != Dynamic) {
        ScalarParamsTraits::PlusEq(
            v[i], delta.template segment<ScalarParamsTraits::Dims>(i * ScalarParamsTraits::Dims));
      } else {
        ScalarParamsTraits::PlusEq(v[i], delta.segment(i, i * ScalarParamsTraits::dims(v[i])));
      }
    }
  }
};

// Trait specialization for std::array
template <typename _Scalar, std::size_t N>
struct params_trait<std::array<_Scalar, N>> {
  using T = typename std::array<_Scalar, N>;
  using Scalar = _Scalar;  // The scalar type
  using ScalarParamsTraits = params_trait<Scalar>;
  static constexpr Index Dims =
      ScalarParamsTraits::Dims == Dynamic
          ? Dynamic
          : N * ScalarParamsTraits::Dims;  // Compile-time parameters dimensions
  // Execution-time parameters dimensions
  static Index dims(const T& v) {
    constexpr int ScalarDims = ScalarParamsTraits::Dims;
    if constexpr (std::is_scalar_v<Scalar> || ScalarDims == 1) {
      return N;
    } else if constexpr (ScalarDims == Dynamic) {
      int d = 0;
      for (std::size_t i = 0; i < N; ++i) d += ScalarParamsTraits::dims(v[i]);
      return d;
    } else {
      return static_cast<Index>(v.size()) * ScalarDims;
    }
  }

  // Cast to a new type, only needed when using automatic differentiation
  template <typename T2>
  static auto cast(const T& v) {
    using Scalar2 =
        std::decay_t<decltype(ScalarParamsTraits::template cast<T2>(std::declval<Scalar>()))>;
    std::array<Scalar2, N> o;
    for (std::size_t i = 0; i < N; ++i) o[i] = ScalarParamsTraits::template cast<T2>(v[i]);
    return o;
  }
  // Define update / manifold
  static void PlusEq(T& v, const auto& delta) {
    for (std::size_t i = 0; i < N; ++i) {
      if constexpr (std::is_scalar_v<Scalar> || ScalarParamsTraits::Dims == 1)
        v[i] += delta[i];
      else if constexpr (ScalarParamsTraits::Dims != Dynamic) {
        ScalarParamsTraits::PlusEq(
            v[i], delta.template segment<ScalarParamsTraits::Dims>(i * ScalarParamsTraits::Dims));
      } else {
        ScalarParamsTraits::PlusEq(v[i], delta.segment(i, i * ScalarParamsTraits::dims(v[i])));
      }
    }
  }
};

// Trait specialization for std::array
template <typename T1, typename T2>
struct params_trait<std::pair<T1, T2>> {
  using T = std::pair<T1, T2>;
  using Scalar = typename params_trait<T1>::Scalar;
  using Scalar1ParamsTraits = params_trait<T1>;
  using Scalar2ParamsTraits = params_trait<T2>;
  // Compile-time parameters dimensions
  static constexpr Index Dims =
      (Scalar1ParamsTraits::Dims == Dynamic || Scalar2ParamsTraits::Dims == Dynamic)
          ? Dynamic
          : Scalar1ParamsTraits::Dims + Scalar2ParamsTraits::Dims;

  // Execution-time parameters dimensions
  static Index dims(const T& v) {
    return Scalar1ParamsTraits::dims(v.first) + Scalar2ParamsTraits::dims(v.second);
  }
  // Cast to a new type, only needed when using automatic differentiation
  template <typename T3>
  static auto cast(const T& v) {
    using Scalar1 =
        std::decay_t<decltype(Scalar1ParamsTraits::template cast<T3>(std::declval<T1>()))>;
    using Scalar2 =
        std::decay_t<decltype(Scalar2ParamsTraits::template cast<T3>(std::declval<T2>()))>;
    std::pair<Scalar1, Scalar2> o{Scalar1ParamsTraits::template cast<T3>(v.first),
                                  Scalar2ParamsTraits::template cast<T3>(v.second)};
    return o;
  }
  // Define update / manifold
  static void PlusEq(T& v, const auto& delta) {
    if constexpr (Scalar1ParamsTraits::Dims == Dynamic)
      Scalar1ParamsTraits::PlusEq(v.first, delta.head(Scalar1ParamsTraits::dims(v.first)));
    else
      Scalar1ParamsTraits::PlusEq(v.first, delta.template head<Scalar1ParamsTraits::Dims>());
    if constexpr (Scalar2ParamsTraits::Dims == Dynamic)
      Scalar2ParamsTraits::PlusEq(v.second, delta.tail(Scalar2ParamsTraits::dims(v.second)));
    else
      Scalar2ParamsTraits::PlusEq(v.first, delta.template tail<Scalar2ParamsTraits::Dims>());
  }
};

/// Return the dynamic dimensions of a parameter type `T`
template <typename T>
inline auto DynDims(const T& x) {
  using ptrait = params_trait<T>;
  if constexpr (ptrait::Dims == Dynamic)
    return ptrait::dims(x);
  else
    return ptrait::Dims;
}

}  // namespace tinyopt::traits
