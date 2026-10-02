// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>
#include <utility>

#include <tinyopt/traits/matrix_traits.h>

namespace tinyopt::traits {

template <typename T, typename = void>
struct params_trait {
  using Scalar = typename T::Scalar;
  static constexpr Index Dims = Dynamic;

  static Index dims(const T &v) { return v.dims(); }

  template <typename T2>
  static auto cast(const T &v) {
    if constexpr (has_static_cast_v<T>)
      return T::template cast<T2>(v);
    else if constexpr (has_cast_v<T>)
      return v.template cast<T2>();
    else
      return T2(v);
  }

  static void PlusEq(T &v, const auto &delta) { v += delta; }
};

template <typename T>
struct params_trait<T, std::void_t<decltype(T::Dims)>> {
  using Scalar = typename T::Scalar;
  static constexpr Index Dims = T::Dims;

  static Index dims(const T &v) { return Dims == Dynamic ? v.dims() : Dims; }

  template <typename T2>
  static auto cast(const T &v) {
    if constexpr (has_static_cast_v<T>)
      return T::template cast<T2>(v);
    else if constexpr (has_cast_v<T>)
      return v.template cast<T2>();
    else
      return T2(v);
  }

  static void PlusEq(T &v, const auto &delta) { v += delta; }
};

template <typename T>
struct params_trait<T, std::enable_if_t<std::is_scalar_v<T>>> {
  using Scalar = T;
  static constexpr Index Dims = 1;

  static constexpr Index dims(const T &) { return 1; }

  template <typename T2>
  static T2 cast(const T &v) {
    return static_cast<T2>(v);
  }

  static void PlusEq(T &v, const auto &delta) { v += delta[0]; }
  static void PlusEq(T &v, const Scalar &delta) { v += delta; }
};

template <typename T>
struct params_trait<T, std::enable_if_t<is_matrix_or_array_v<T>>> {
  using Scalar = typename T::Scalar;
  static constexpr int ColsAtCompileTime = T::ColsAtCompileTime;
  static constexpr Index Dims =
      (T::RowsAtCompileTime == Dynamic || T::ColsAtCompileTime == Dynamic)
          ? Dynamic
          : T::RowsAtCompileTime * T::ColsAtCompileTime;

  static Index dims(const T &m) { return m.size(); }

  template <typename T2>
  static auto cast(const T &v) {
    return v.template cast<T2>().eval();
  }

  static void PlusEq(T &v, const auto &delta) {
    if constexpr (Dims == Dynamic) assert(delta.rows() == (int)v.size());
    if constexpr (T::ColsAtCompileTime == 1)
      v += delta;
    else
      v += delta.reshaped(v.rows(), v.cols());
  }
};

template <typename T>
struct params_trait<T, std::enable_if_t<is_sparse_matrix_v<T>>> {
  using Scalar = typename T::Scalar;
  static constexpr Index Dims = Dynamic;

  static Index dims(const T &m) { return m.size(); }

  template <typename T2>
  static auto cast(const T &v) {
    return v.template cast<T2>().eval();
  }

  static void PlusEq(T &v, const auto &delta) {
    if constexpr (Dims == Dynamic) assert(delta.rows() == (int)v.size());
    if constexpr (T::ColsAtCompileTime == 1)
      v += delta;
    else
      v += delta.reshaped(v.rows(), v.cols());
  }
};

template <typename _Scalar>
struct params_trait<std::vector<_Scalar>> {
  using T = typename std::vector<_Scalar>;
  using Scalar = _Scalar;
  using ScalarParamsTraits = params_trait<Scalar>;
  static constexpr Index Dims = Dynamic;

  static Index dims(const T &v) {
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

  template <typename T2>
  static auto cast(const T &v) {
    using Scalar2 = std::decay_t<decltype(ScalarParamsTraits::template cast<T2>(std::declval<Scalar>()))>;
    std::vector<Scalar2> o;
    o.reserve(v.size());
    for (auto &x : v) o.emplace_back(ScalarParamsTraits::template cast<T2>(x));
    return o;
  }

  static void PlusEq(T &v, const auto &delta) {
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

template <typename _Scalar, std::size_t N>
struct params_trait<std::array<_Scalar, N>> {
  using T = typename std::array<_Scalar, N>;
  using Scalar = _Scalar;
  using ScalarParamsTraits = params_trait<Scalar>;
  static constexpr Index Dims =
      ScalarParamsTraits::Dims == Dynamic ? Dynamic : N * ScalarParamsTraits::Dims;

  static Index dims(const T &v) {
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

  template <typename T2>
  static auto cast(const T &v) {
    using Scalar2 = std::decay_t<decltype(ScalarParamsTraits::template cast<T2>(std::declval<Scalar>()))>;
    std::array<Scalar2, N> o;
    for (std::size_t i = 0; i < N; ++i) o[i] = ScalarParamsTraits::template cast<T2>(v[i]);
    return o;
  }

  static void PlusEq(T &v, const auto &delta) {
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

template <typename T1, typename T2>
struct params_trait<std::pair<T1, T2>> {
  using T = std::pair<T1, T2>;
  using Scalar = typename params_trait<T1>::Scalar;
  using Scalar1ParamsTraits = params_trait<T1>;
  using Scalar2ParamsTraits = params_trait<T2>;
  static constexpr Index Dims =
      (Scalar1ParamsTraits::Dims == Dynamic || Scalar2ParamsTraits::Dims == Dynamic)
          ? Dynamic
          : Scalar1ParamsTraits::Dims + Scalar2ParamsTraits::Dims;

  static Index dims(const T &v) {
    return Scalar1ParamsTraits::dims(v.first) + Scalar2ParamsTraits::dims(v.second);
  }

  template <typename T3>
  static auto cast(const T &v) {
    using Scalar1 = std::decay_t<decltype(Scalar1ParamsTraits::template cast<T3>(std::declval<T1>()))>;
    using Scalar2 = std::decay_t<decltype(Scalar2ParamsTraits::template cast<T3>(std::declval<T2>()))>;
    std::pair<Scalar1, Scalar2> o{Scalar1ParamsTraits::template cast<T3>(v.first),
                                  Scalar2ParamsTraits::template cast<T3>(v.second)};
    return o;
  }

  static void PlusEq(T &v, const auto &delta) {
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

template <typename T>
inline auto DynDims(const T &x) {
  using ptrait = params_trait<T>;
  if constexpr (ptrait::Dims == Dynamic)
    return ptrait::dims(x);
  else
    return ptrait::Dims;
}

}  // namespace tinyopt::traits
