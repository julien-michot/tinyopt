// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cassert>
#include <type_traits>
#include <utility>
#include <vector>

#include <tinyopt/traits/matrix_traits.h>

namespace tinyopt {

namespace traits {

namespace detail {

template <typename T, typename = void>
struct get_param_dims : std::integral_constant<Index, Dynamic> {};
template <typename T>
struct get_param_dims<T, std::void_t<decltype(T::Dims)>> : std::integral_constant<Index, T::Dims> {
};

template <typename T>
inline constexpr Index param_dims_v = get_param_dims<T>::value;

template <typename T, typename = void>
struct param_scalar {
  using type = double;
};
template <typename T>
struct param_scalar<T, std::void_t<typename T::Scalar>> {
  using type = typename T::Scalar;
};

template <typename T, typename = void>
struct has_dims_method : std::false_type {};
template <typename T>
struct has_dims_method<T, std::void_t<decltype(std::declval<const T &>().dims())>>
    : std::true_type {};
template <typename T>
inline constexpr bool has_dims_method_v = has_dims_method<T>::value;

template <typename T, typename = void>
struct has_size_method : std::false_type {};
template <typename T>
struct has_size_method<T, std::void_t<decltype(std::declval<const T &>().size())>>
    : std::true_type {};
template <typename T>
inline constexpr bool has_size_method_v = has_size_method<T>::value;

template <typename T, typename Delta, typename = void>
struct has_plus_eq_static : std::false_type {};
template <typename T, typename Delta>
struct has_plus_eq_static<
    T, Delta, std::void_t<decltype(T::PlusEq(std::declval<T &>(), std::declval<const Delta &>()))>>
    : std::true_type {};

template <typename T, typename = void>
struct has_member_locked : std::false_type {};
template <typename T>
struct has_member_locked<
    T, std::void_t<decltype(std::declval<const std::remove_cvref_t<T> &>().locked())>>
    : std::true_type {};

template <typename T, typename = void>
struct has_nonconst_member_locked : std::false_type {};
template <typename T>
struct has_nonconst_member_locked<
    T, std::void_t<decltype(std::declval<std::remove_cvref_t<T> &>().locked())>> : std::true_type {
};

template <typename T, typename = void>
struct has_static_locked : std::false_type {};
template <typename T>
struct has_static_locked<T, std::void_t<decltype(std::remove_cvref_t<T>::locked())>>
    : std::true_type {};

template <typename T>
inline constexpr bool has_member_locked_v = has_member_locked<T>::value;

template <typename T>
inline constexpr bool has_nonconst_member_locked_v = has_nonconst_member_locked<T>::value;

template <typename T>
inline constexpr bool has_static_locked_v = has_static_locked<T>::value;

}  // namespace detail

template <typename T, typename = void>
struct params_trait {
  using Scalar = typename detail::param_scalar<T>::type;
  static constexpr Index Dims = detail::param_dims_v<T>;

  static Index dims(const T &v) {
    if constexpr (Dims != Dynamic) {
      return Dims;
    } else if constexpr (detail::has_dims_method_v<T>) {
      return v.dims();
    } else if constexpr (detail::has_size_method_v<T>) {
      return v.size();
    } else {
      return Dims;
    }
  }

  template <typename T2>
  static auto cast(const T &v) {
    if constexpr (has_static_cast_v<T>)
      return T::template cast<T2>(v);
    else if constexpr (has_cast_v<T>)
      return v.template cast<T2>();
    else
      return T2(v);
  }

  template <typename Delta>
  static void PlusEq(T &v, const Delta &delta) {
    if constexpr (detail::has_plus_eq_static<T, Delta>::value)
      T::PlusEq(v, delta);
    else
      v += delta;
  }

  static auto locked(const T &v)
    requires(detail::has_member_locked_v<T> || detail::has_nonconst_member_locked_v<T> ||
             detail::has_static_locked_v<T>)
  {
    if constexpr (detail::has_member_locked<T>::value) {
      return v.locked();
    } else if constexpr (detail::has_nonconst_member_locked<T>::value) {
      return const_cast<T &>(v).locked();
    } else {
      return T::locked();
    }
  }

  static auto locked()
    requires(detail::has_static_locked<T>::value)
  {
    return T::locked();
  }
};

namespace detail {

template <typename T, typename = void>
struct has_trait_locked : std::false_type {};
template <typename T>
struct has_trait_locked<
    T, std::enable_if_t<!std::is_scalar_v<std::remove_cvref_t<T>> &&
                            !is_matrix_or_array_v<std::remove_cvref_t<T>> &&
                            !is_sparse_matrix_v<std::remove_cvref_t<T>>,
                        std::void_t<decltype(params_trait<std::remove_cvref_t<T>>::locked(
                            std::declval<const std::remove_cvref_t<T> &>()))>>> : std::true_type {};

template <typename T, typename = void>
struct has_trait_static_locked : std::false_type {};
template <typename T>
struct has_trait_static_locked<
    T, std::enable_if_t<!std::is_scalar_v<std::remove_cvref_t<T>> &&
                            !is_matrix_or_array_v<std::remove_cvref_t<T>> &&
                            !is_sparse_matrix_v<std::remove_cvref_t<T>>,
                        std::void_t<decltype(params_trait<std::remove_cvref_t<T>>::locked())>>>
    : std::true_type {};
}  // namespace detail
template <typename T>
concept has_locked = requires(const std::remove_cvref_t<T> &x) {
  { params_trait<std::remove_cvref_t<T>>::locked(x) };
} || requires {
  { params_trait<std::remove_cvref_t<T>>::locked() };
} || requires(const std::remove_cvref_t<T> &x) {
  { x.locked() };
} || requires(std::remove_cvref_t<T> &x) {
  { x.locked() };
} || requires {
  { std::remove_cvref_t<T>::locked() };
};

// (Optional) Keep the variable template if it's part of your public API
template <typename T>
inline constexpr bool has_locked_v = has_locked<T>;

template <typename T>
  requires(has_locked_v<T>)
inline auto locked(const T &x) {
  using PureT = std::remove_cvref_t<T>;
  if constexpr (requires(const PureT &val) { params_trait<PureT>::locked(val); }) {
    return params_trait<PureT>::locked(x);
  } else if constexpr (requires { params_trait<PureT>::locked(); }) {
    return params_trait<PureT>::locked();
  } else if constexpr (requires(const PureT &val) { val.locked(); }) {
    return x.locked();
  } else if constexpr (requires(PureT &val) { val.locked(); }) {
    return const_cast<PureT &>(x).locked();
  } else {
    return PureT::locked();
  }
}

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
  using PureT = std::remove_cvref_t<T>;
  using Scalar = typename PureT::Scalar;
  static constexpr int ColsAtCompileTime = PureT::ColsAtCompileTime;
  static constexpr Index Dims =
      (PureT::RowsAtCompileTime == Dynamic || PureT::ColsAtCompileTime == Dynamic)
          ? Dynamic
          : PureT::RowsAtCompileTime * PureT::ColsAtCompileTime;

  static Index dims(const T &m) { return m.size(); }

  template <typename T2>
  static auto cast(const T &v) {
    return v.template cast<T2>().eval();
  }

  static void PlusEq(T &v, const auto &delta) {
    if constexpr (Dims == Dynamic) assert(delta.rows() == (int)v.size());
    if constexpr (PureT::ColsAtCompileTime == 1)
      v += delta;
    else
      v += delta.reshaped(v.rows(), v.cols());
  }
};

template <typename T>
struct params_trait<T, std::enable_if_t<is_sparse_matrix_v<T>>> {
  using PureT = std::remove_cvref_t<T>;
  using Scalar = typename PureT::Scalar;
  static constexpr Index Dims = Dynamic;

  static Index dims(const T &m) { return m.size(); }

  template <typename T2>
  static auto cast(const T &v) {
    return v.template cast<T2>().eval();
  }

  static void PlusEq(T &v, const auto &delta) {
    assert(delta.rows() == (int)v.size());
    if constexpr (T::ColsAtCompileTime == 1)
      v += delta;
    else
      v += delta.reshaped(v.rows(), v.cols());
  }
};

template <typename _Scalar>
struct params_trait<std::vector<_Scalar>> {
  using T = std::vector<_Scalar>;
  using Scalar = _Scalar;
  using ScalarParamsTraits = params_trait<Scalar>;
  static constexpr Index Dims = Dynamic;

  static Index dims(const T &v) {
    constexpr int ScalarDims = ScalarParamsTraits::Dims;
    if constexpr (std::is_scalar_v<Scalar> || ScalarDims == 1) {
      return static_cast<Index>(v.size());
    } else if constexpr (ScalarDims == Dynamic) {
      Index d = 0;
      for (std::size_t i = 0; i < v.size(); ++i) d += ScalarParamsTraits::dims(v[i]);
      return d;
    } else {
      return static_cast<Index>(v.size()) * ScalarDims;
    }
  }

  template <typename T2>
  static auto cast(const T &v) {
    using Scalar2 =
        std::decay_t<decltype(ScalarParamsTraits::template cast<T2>(std::declval<Scalar>()))>;
    std::vector<Scalar2> o;
    o.reserve(v.size());
    for (const auto &x : v) o.emplace_back(ScalarParamsTraits::template cast<T2>(x));
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

  static auto locked(const T &v)
    requires(detail::has_member_locked_v<T> || detail::has_nonconst_member_locked_v<T> ||
             detail::has_static_locked_v<T>)
  {
    std::vector<Index> res;
    Index offset = 0;
    for (const auto &item : v) {
      auto l = traits::locked(item);
      for (auto idx : l) res.push_back(offset + idx);
      offset += ScalarParamsTraits::dims(item);
    }
    return res;
  }
};

template <typename _Scalar, std::size_t N>
struct params_trait<std::array<_Scalar, N>> {
  using T = std::array<_Scalar, N>;
  using Scalar = _Scalar;
  using ScalarParamsTraits = params_trait<Scalar>;
  static constexpr Index Dims = ScalarParamsTraits::Dims == Dynamic
                                    ? Dynamic
                                    : static_cast<Index>(N) * ScalarParamsTraits::Dims;

  static Index dims(const T &v) {
    constexpr int ScalarDims = ScalarParamsTraits::Dims;
    if constexpr (std::is_scalar_v<Scalar> || ScalarDims == 1) {
      return static_cast<Index>(N);
    } else if constexpr (ScalarDims == Dynamic) {
      Index d = 0;
      for (std::size_t i = 0; i < N; ++i) d += ScalarParamsTraits::dims(v[i]);
      return d;
    } else {
      return static_cast<Index>(v.size()) * ScalarDims;
    }
  }

  template <typename T2>
  static auto cast(const T &v) {
    using Scalar2 =
        std::decay_t<decltype(ScalarParamsTraits::template cast<T2>(std::declval<Scalar>()))>;
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

  static auto locked(const T &v)
    requires(has_locked_v<Scalar>)
  {
    std::vector<Index> res;
    Index offset = 0;
    for (std::size_t i = 0; i < N; ++i) {
      auto l = traits::locked(v[i]);
      for (auto idx : l) res.push_back(offset + idx);
      offset += ScalarParamsTraits::dims(v[i]);
    }
    return res;
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
    using Scalar1 =
        std::decay_t<decltype(Scalar1ParamsTraits::template cast<T3>(std::declval<T1>()))>;
    using Scalar2 =
        std::decay_t<decltype(Scalar2ParamsTraits::template cast<T3>(std::declval<T2>()))>;
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
      Scalar2ParamsTraits::PlusEq(v.second, delta.template tail<Scalar2ParamsTraits::Dims>());
  }

  static auto locked(const T &v)
    requires(has_locked_v<T1> || has_locked_v<T2>)
  {
    std::vector<Index> res;
    if constexpr (has_locked_v<T1>) {
      auto l1 = traits::locked(v.first);
      for (auto idx : l1) res.push_back(idx);
    }
    if constexpr (has_locked_v<T2>) {
      auto l2 = traits::locked(v.second);
      const Index offset = Scalar1ParamsTraits::dims(v.first);
      for (auto idx : l2) res.push_back(offset + idx);
    }
    return res;
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

}  // namespace traits

using traits::has_locked_v;
using traits::locked;

}  // namespace tinyopt
