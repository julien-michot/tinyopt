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

#include <tinyopt/math.h>
#include <tinyopt/traits.h>
#include <cmath>
#include <type_traits>

namespace tinyopt::losses {

/**
 * @name Classical Norms
 * @{
 */

/// Return the squared L2 norm of a vector or scalar (a.k.a Sum of Squares)
template <typename T>
auto SquaredL2(const T &x) {
  constexpr bool IsMatrix = traits::is_matrix_or_array_v<T> || traits::is_sparse_matrix_v<T>;
  if constexpr (traits::is_pair_v<T>) {  // pair
    return SquaredL2(x.first, x.second);
  } else if constexpr (IsMatrix) {  // scalar
    return x.squaredNorm();
  } else {  // scalar
    return x * x;
  }
}

/// Return the squared L2 norm of a vector or scalar and its jacobian  (x.t())
template <typename T, typename ExportJ>
auto SquaredL2(const T &x, const ExportJ &Jx_or_bool) {
  constexpr bool IsMatrix = traits::is_matrix_or_array_v<T> || traits::is_sparse_matrix_v<T>;
  const auto l = SquaredL2(x);
  if constexpr (IsMatrix) {
    using Scalar = typename traits::params_trait<T>::Scalar;
    if constexpr (std::is_same_v<ExportJ, bool>)
      return std::make_pair(l, (Scalar(2.0) * x.transpose()).eval());
    else
      return std::make_pair(l, (Scalar(2.0) * x.transpose() * Jx_or_bool).eval());
  } else {  // scalar
    return std::make_pair(l, x);
  }
}

/// Return L2 norm of a vector or scalar
template <typename T>
auto L2(const T &x) {
  constexpr bool IsMatrix = traits::is_matrix_or_array_v<T> || traits::is_sparse_matrix_v<T>;
  if constexpr (traits::is_pair_v<T>) {  // pair
    return L2(x.first, x.second);
  } else if constexpr (IsMatrix) {  // scalar
    return x.norm();
  } else {  // scalar
    using std::abs;
    return abs(x);  // same as sqrt(x*x);
  }
}

/// Return L2 norm of a vector or scalar and its jacobian (x.t() / ||x||)
template <typename T, typename ExportJ>
auto L2(const T &x, const ExportJ &Jx_or_bool) {
  constexpr bool IsMatrix = traits::is_matrix_or_array_v<T> || traits::is_sparse_matrix_v<T>;
  const auto l = L2(x);
  if constexpr (IsMatrix) {
    const auto J = l > FloatEpsilon() ? (x / l).eval() : x;
    if constexpr (std::is_same_v<ExportJ, bool>)
      return std::make_pair(l, J.transpose().eval());
    else
      return std::make_pair(l, (J.transpose() * Jx_or_bool).eval());
  } else {  // scalar
    return std::make_pair(l, x);
  }
}

/// Return L1 norm of a vector or scalar
template <typename T>
auto L1(const T &x) {
  constexpr bool IsMatrix = traits::is_matrix_or_array_v<T> || traits::is_sparse_matrix_v<T>;
  if constexpr (traits::is_pair_v<T>) {  // pair
    return L1(x.first, x.second);
  } else if constexpr (IsMatrix) {  // scalar
    return x.template lpNorm<1>();
  } else {  // scalar
    using std::abs;
    return abs(x);
  }
}

/// Return L1 norm of a vector or scalar and its jacobian
template <typename T, typename ExportJ>
auto L1(const T &x, const ExportJ &Jx_or_bool) {
  constexpr bool IsMatrix = traits::is_matrix_or_array_v<T> || traits::is_sparse_matrix_v<T>;
  const auto l = L1(x);
  if constexpr (IsMatrix) {
    using Scalar = typename traits::params_trait<T>::Scalar;
    constexpr int DimsJ = traits::params_trait<T>::Dims;
    if constexpr (std::is_same_v<ExportJ, bool>)
      return std::make_pair(l, RowVector<Scalar, DimsJ>(x.array().sign()));
    else
      return std::make_pair(l, (x.array().sign() * Jx_or_bool).eval());
  } else {  // scalar
    return std::make_pair(l, 1);
  }
}

/// Return L infinity norm of a vector or scalar
template <typename T>
auto Linf(const T &x) {
  constexpr bool IsMatrix = traits::is_matrix_or_array_v<T> || traits::is_sparse_matrix_v<T>;
  if constexpr (traits::is_pair_v<T>) {  // pair
    return Linf(x.first, x.second);
  } else if constexpr (IsMatrix) {
    return x.template lpNorm<Infinity>();
  } else {  // scalar
    using std::abs;
    return abs(x);
  }
}

/// Return L infinity norm of a vector or scalar
template <typename T, typename ExportJ>
auto Linf(const T &x, const ExportJ &Jx_or_bool) {
  constexpr bool IsMatrix = traits::is_matrix_or_array_v<T> || traits::is_sparse_matrix_v<T>;
  if constexpr (IsMatrix) {
    using Scalar = typename traits::params_trait<T>::Scalar;
    constexpr int DimsJ = traits::params_trait<T>::Dims;
    RowVector<Scalar, DimsJ> J(1, x.rows());
    J.setZero();
    int max_idx;
    const auto l = x.cwiseAbs().maxCoeff(&max_idx);
    J[max_idx] = x[max_idx] >= 0 ? 1 : -1;
    if constexpr (std::is_same_v<ExportJ, bool>)
      return std::make_pair(l, J);
    else
      return std::make_pair(l, (J * Jx_or_bool).eval());
  } else {  // scalar
    if (x >= 0)
      return std::make_pair(x, T(1.0));
    else
      return std::make_pair(-x, T(-1.0));
  }
}

/** @} */

/**
 * @name Mahalanobis Distances // TODO MOVE to distances.h
 * @{
 */

/// Return scaled residuals (and its jacobian J as a option) when applying a
/// Mahalanobis norm when given residuals and a covariance matrix (Upper filled at least)
template <typename Derived, typename DerivedC, typename Jac_t = std::nullptr_t>
Vector<typename Derived::Scalar, Derived::RowsAtCompileTime> Mah(const MatrixBase<Derived> &res,
                                                                 const MatrixBase<DerivedC> &cov,
                                                                 Jac_t *J = nullptr) {
  using Scalar = typename Derived::Scalar;
  static constexpr int Dims = Derived::RowsAtCompileTime;
  using Mat = Matrix<Scalar, Dims, Dims>;
  const auto chol = Eigen::SelfAdjointView<const Mat, Upper>(cov).llt();
  const Mat L = chol.matrixL();  // L
  if constexpr (!std::is_same_v<Jac_t, std::nullptr_t>)
    if (J) *J = (L.template triangularView<Lower>().solve(*J)).eval();  // J must be filled!
  return L.template triangularView<Lower>().solve(res);
}

/// Return scaled residuals (and its jacobian J as a option) when applying a
/// Mahalanobis norm when given residuals and a upper triangular information matrix
template <typename Derived, typename DerivedC, typename Jac_t = std::nullptr_t>
Vector<typename Derived::Scalar, Derived::RowsAtCompileTime> MahInfoU(
    const MatrixBase<Derived> &res, const MatrixBase<DerivedC> &L, Jac_t *J = nullptr) {
  if constexpr (!std::is_same_v<Jac_t, std::nullptr_t>)
    if (J) *J = (L.template triangularView<Upper>() * (*J)).eval();
  return L.template triangularView<Upper>() * res;
}

/// Return scaled residuals (and its jacobian J as a option) when applying a
/// Mahalanobis norm when given residuals and a vector of standard deviations
template <typename Derived, typename Derived2, typename Jac_t = std::nullptr_t>
auto MahDiag(const MatrixBase<Derived> &res, const MatrixBase<Derived2> &stdevs,
             Jac_t *J = nullptr) {
  if constexpr (!std::is_same_v<Jac_t, std::nullptr_t>)
    if (J) J->noalias() = (J->array().colwise() / stdevs.array()).matrix();
  return (res.array() / stdevs.array()).eval();
}

/// Return scaled residuals (and its jacobian J as a option) when applying a scale to the residuals
template <typename Derived, typename Jac_t = std::nullptr_t>
auto Iso(const MatrixBase<Derived> &res, typename Derived::Scalar &scale, Jac_t *J = nullptr) {
  if constexpr (!std::is_same_v<Jac_t, std::nullptr_t>)
    if (J) *J *= scale;
  return (res * scale).eval();
}

/** @} */

}  // namespace tinyopt::losses
