// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tinyopt/math/block_sparse_matrix.h>
#include <tinyopt/traits/callable_traits.h>
#include <tinyopt/traits/matrix_traits.h>
#include <tinyopt/traits/parameter_traits.h>
#include <tinyopt/traits/params_pack.h>
#include <tinyopt/traits/type_traits.h>

namespace tinyopt::losses {
template <typename ResidualT, typename LossTag>
struct RobustResidual;
}

namespace tinyopt::traits {

template <typename T>
struct is_robust_residual : std::false_type {};
template <typename ResidualT, typename LossTag>
struct is_robust_residual<losses::RobustResidual<ResidualT, LossTag>> : std::true_type {};
template <typename T>
inline constexpr bool is_robust_residual_v = is_robust_residual<std::remove_cvref_t<T>>::value;

template <typename Scalar_, int... BlockSizes>
struct params_trait<BlockSparseMatrix<Scalar_, BlockSizes...>> {
  using Scalar = Scalar_;
  static constexpr Index Dims = Dynamic;

  static Index dims(const BlockSparseMatrix<Scalar_, BlockSizes...> &matrix) {
    return matrix.rows();
  }

  static auto norm2(const BlockSparseMatrix<Scalar_, BlockSizes...> &matrix) {
    return matrix.squaredNorm();
  }
};

}  // namespace tinyopt::traits
