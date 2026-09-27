// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tinyopt/types.h>
#include <tinyopt/traits/type_traits.h>

namespace tinyopt::traits {

// Trait to check if a type is a Sparse Matrix.
template <typename T>
struct is_sparse_matrix : std::false_type {};
template <typename T>
struct is_sparse_matrix<SparseMatrix<T>> : std::true_type {};

template <typename T>
inline constexpr bool is_sparse_matrix_v = is_sparse_matrix<std::decay_t<T>>::value;

// Trait to check if a type is a Matrix/Vector.
template <typename T, typename = void>
struct is_matrix_or_array : std::false_type {};

template <typename T>
struct is_matrix_or_array<
    T,
    std::void_t<decltype(std::declval<const std::remove_cvref_t<T>&>().rows()),
               decltype(std::declval<const std::remove_cvref_t<T>&>().cols()),
               typename std::remove_cvref_t<T>::Scalar>>
    : std::bool_constant<!is_sparse_matrix<std::remove_cvref_t<T>>::value> {};

template <typename T>
inline constexpr bool is_matrix_or_array_v = is_matrix_or_array<std::remove_cvref_t<T>>::value;

template <typename T>
inline constexpr bool is_matrix_or_scalar_v =
    (std::is_scalar_v<T> && !std::is_same_v<T, bool>) || is_sparse_matrix_v<T> ||
    is_matrix_or_array_v<T>;

}  // namespace tinyopt::traits
