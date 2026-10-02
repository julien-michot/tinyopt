// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>
#include <utility>
#include <ostream>
#include <tuple>
#include <array>
#include <vector>

namespace tinyopt::traits {

// Check whether a type 'T' or '&T' is nullptr_t.
template <typename T>
struct is_nullptr_t : std::is_same<std::decay_t<T>, std::nullptr_t> {};
template <typename T>
inline constexpr bool is_nullptr_v = is_nullptr_t<T>::value;

// Check whether a type 'T' or '&T' is a bool.
template <typename T>
struct is_bool : std::is_same<std::decay_t<T>, bool> {};
template <typename T>
inline constexpr bool is_bool_v = is_bool<T>::value;

// Check whether a type 'T' or '&T' is a scalar.
template <typename T>
struct is_scalar : std::is_scalar<std::decay_t<T>> {};
template <typename T>
inline constexpr bool is_scalar_v = is_scalar<T>::value;

// Trait to detect std::pair.
template <typename T>
struct is_pair : std::false_type {};
template <typename T, typename U>
struct is_pair<std::pair<T, U>> : std::true_type {};
template <typename T>
inline constexpr bool is_pair_v = is_pair<std::remove_cvref_t<T>>::value;

// Trait to detect std::tuple.
template <typename T>
struct is_tuple : std::false_type {};
template <typename... Ts>
struct is_tuple<std::tuple<Ts...>> : std::true_type {};
template <typename T>
inline constexpr bool is_tuple_v = is_tuple<std::remove_cvref_t<T>>::value;

// Trait to detect if a type is streamable.
template <typename T, typename = void>
struct is_streamable : std::false_type {};

template <typename T>
struct is_streamable<T, std::void_t<decltype(std::declval<std::ostream&>() << std::declval<T>())>>
    : std::true_type {};

template <typename T>
inline constexpr bool is_streamable_v = is_streamable<T>::value;

// Trait to check if a type has a cast method.
template <typename T, typename = void>
struct has_cast : std::false_type {};

template <typename T>
struct has_cast<T, std::void_t<decltype(std::declval<const T>().template cast<int>())>>
    : std::true_type {};

template <typename T>
inline constexpr bool has_cast_v = has_cast<T>::value;

template <typename T, typename = void>
struct has_static_cast : std::false_type {};

template <typename T>
struct has_static_cast<T,
                      std::void_t<decltype(T::template cast<float>(std::declval<const T&>()))>>
    : std::true_type {};

template <typename T>
inline constexpr bool has_static_cast_v = has_static_cast<T>::value;

}  // namespace tinyopt::traits
