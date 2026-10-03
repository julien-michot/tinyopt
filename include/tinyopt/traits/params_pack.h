// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>
#include <type_traits>
#include <utility>

#include <tinyopt/params_pack.h>
#include <tinyopt/traits/parameter_traits.h>

namespace tinyopt::traits {

namespace detail {

template <typename... Ts>
struct PackDims;

template <typename T>
struct PackDims<T> : std::integral_constant<Index, params_trait<std::remove_cvref_t<T>>::Dims> {};

template <typename T, typename U, typename... Ts>
struct PackDims<T, U, Ts...> {
 private:
  static constexpr Index First = params_trait<std::remove_cvref_t<T>>::Dims;
  static constexpr Index Rest = PackDims<U, Ts...>::value;

 public:
  static constexpr Index value = First == Dynamic || Rest == Dynamic ? Dynamic : First + Rest;
};

template <typename T, typename = void>
struct IsJetScalar : std::false_type {};

template <typename T>
struct IsJetScalar<T, std::void_t<decltype(std::declval<T>().a), decltype(std::declval<T>().v)>>
    : std::true_type {};

template <typename T, typename Delta>
void ApplyPackDelta(T &param, Index &offset, const Delta &delta) {
  using Param = params_trait<std::remove_cvref_t<T>>;
  if constexpr (std::is_scalar_v<std::remove_cvref_t<T>> ||
                IsJetScalar<std::remove_cvref_t<T>>::value) {
    Param::PlusEq(param, delta[offset]);
    ++offset;
  } else if constexpr (Param::Dims == Dynamic) {
    const Index dims = Param::dims(param);
    Param::PlusEq(param, delta.segment(offset, dims));
    offset += dims;
  } else {
    Param::PlusEq(param, delta.template segment<Param::Dims>(offset));
    offset += Param::Dims;
  }
}

}  // namespace detail

template <typename... Ts>
struct params_trait<::tinyopt::ParamsPack<Ts...>> {
  using Scalar = std::common_type_t<typename params_trait<std::remove_cvref_t<Ts>>::Scalar...>;
  static constexpr Index Dims = detail::PackDims<Ts...>::value;

  static Index dims(const ::tinyopt::ParamsPack<Ts...> &pack) {
    return std::apply(
        [](const auto &...params) {
          return (params_trait<std::remove_cvref_t<decltype(params)>>::dims(params) + ...);
        },
        pack.values);
  }

  template <typename T2>
  static auto cast(const ::tinyopt::ParamsPack<Ts...> &pack) {
    return std::apply(
        [](const auto &...params) {
          return ::tinyopt::ParamsPack<std::decay_t<decltype(
              params_trait<std::remove_cvref_t<decltype(params)>>::template cast<T2>(params))>...>(
              params_trait<std::remove_cvref_t<decltype(params)>>::template cast<T2>(params)...);
        },
        pack.values);
  }

  static void PlusEq(::tinyopt::ParamsPack<Ts...> &pack, const auto &delta) {
    Index offset = 0;
    std::apply([&](auto &...params) { (detail::ApplyPackDelta(params, offset, delta), ...); },
               pack.values);
  }
};

}  // namespace tinyopt::traits