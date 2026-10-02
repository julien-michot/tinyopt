// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tinyopt/traits/matrix_traits.h>

namespace tinyopt::traits {

template <typename Func, typename Param>
inline constexpr bool is_scalar_cost_v =
    std::is_invocable_v<Func, const Param &> &&
    std::is_scalar_v<std::invoke_result_t<Func, const Param &>>;

template <typename Func, typename Param, typename Scalar, Index Dims>
inline constexpr bool accepts_gradient_v =
    std::is_invocable_v<Func, const Param &, Vector<Scalar, Dims> &>;

template <typename Func, typename Param, typename Scalar, Index Dims>
inline constexpr bool accepts_hessian_v =
    std::is_invocable_v<Func, const Param &, Vector<Scalar, Dims> &, Matrix<Scalar, Dims, Dims> &>;

}  // namespace tinyopt::traits
