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
//
// Here are common methods to scale residuals, e.g. Mahalanobis, Huber, GemanMcClure, ...
//
// Example of usage:
//
//   Vec3f res = ...;
//   Mat3f cov = ...;
//
//   Vec3f scaled_res = Mah(res, cov); // scaled residuals
// or
//   Vec3f scaled_res = Mah(res);
//
// or
//   Mat3f J = Mat3f::Identity(); // when passing a jacobian, make sure it is initialized
//                                // since it will be multiplied
//   Vec3f scaled_res = Mah(res, &J);

#pragma once

#include <cstddef>

#include <tinyopt/math.h>
#include <tinyopt/traits.h>

namespace tinyopt::loss {

/// Return scaled residuals (and jacobian J as a option) when applying a L2 norm
/// @note You probably don't need to call this function, it's here just as an example.
template <typename T, typename Jac_t = std::nullptr_t>
auto L2(const T &res, Jac_t *J = nullptr) {
  if constexpr (!std::is_same_v<Jac_t, std::nullptr_t>)
    if (J) J->setIdentity();
  return res;  // no scaling
}

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

}  // namespace tinyopt::loss
