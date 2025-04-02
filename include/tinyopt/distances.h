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

#include <cstddef>
#include <type_traits>

#include <tinyopt/math.h>
#include <tinyopt/traits.h>

namespace tinyopt::distances {

/// Compute the Euclidean L2 distance between `a` and `b`
template <typename TA, typename TB, typename Jac_t = std::nullptr_t>
auto Euclidean(const TA &a, const TB &b, Jac_t *Ja = nullptr, Jac_t *Jb = nullptr) {
  if constexpr (std::is_scalar_v<TA>) {
    if constexpr (!std::is_same_v<Jac_t, std::nullptr_t>) {
      if (Ja) *Ja = 1;
      if (Jb) *Jb = -1;
    }
    return std::abs(a - b);
  } else {
    constexpr double eps2 = FloatEpsilon2<typename TA::Scalar>();
    const auto d = (a - b).eval();
    if constexpr (!std::is_same_v<Jac_t, std::nullptr_t>) {
      const auto dn2 = d.squaredNorm();
      if (dn2 > eps2) {
        if (Ja) *Ja = d / std::sqrt(dn2);
        if (Jb) *Jb = -d / std::sqrt(dn2);
      } else {
        if (Ja) *Ja = Jac_t::Zero();
        if (Jb) *Jb = Jac_t::Zero();
      }
    }
    return d.norm();
  }
}

/// Compute the Manhattan distance between `a` and `b`
template <typename TA, typename TB, typename Jac_t = std::nullptr_t>
auto Manhattan(const TA &a, const TB &b, Jac_t *Ja = nullptr, Jac_t *Jb = nullptr) {
  if constexpr (std::is_scalar_v<TA>) {
    const auto d = std::abs(a - b);
    if constexpr (!std::is_same_v<Jac_t, std::nullptr_t>) {
      if (Ja) *Ja = (a > b) ? 1 : ((a < b) ? -1 : 0);
      if (Jb) *Jb = (a > b) ? -1 : ((a < b) ? 1 : 0);
    }
    return d;
  } else {
    const auto d = (a - b).cwiseAbs().sum();
    if constexpr (!std::is_same_v<Jac_t, std::nullptr_t>) {
      using Scalar = typename TA::Scalar;
      if (Ja)
        *Ja = (a - b).unaryExpr([](Scalar x) { return (x > 0) ? 1.0 : ((x < 0) ? -1.0 : 0.0); });
      if (Jb)
        *Jb = -(a - b).unaryExpr([](Scalar x) { return (x > 0) ? 1.0 : ((x < 0) ? -1.0 : 0.0); });
    }
    return d;
  }
}

/// Compute the cosine distance between `a` and `b`
template <typename TA, typename TB, typename Jac_t = std::nullptr_t>
auto Cosine(const TA &a, const TB &b, Jac_t *Ja = nullptr, Jac_t *Jb = nullptr) {
  if constexpr (std::is_scalar_v<TA>) {
    if constexpr (!std::is_same_v<Jac_t, std::nullptr_t>) {
      if (Ja) *Ja = 0;
      if (Jb) *Jb = 0;
    }
    return 0.0;
  } else {
    using Scalar = typename TA::Scalar;
    constexpr double eps2 = FloatEpsilon2<Scalar>();
    const auto a_norm = a.norm();
    const auto b_norm = b.norm();
    if (a_norm * b_norm < eps2) {
      if constexpr (!std::is_same_v<Jac_t, std::nullptr_t>) {
        if (Ja) *Ja = Jac_t::Zero();
        if (Jb) *Jb = Jac_t::Zero();
      }
      return Scalar(0.0);
    } else {
      const auto ab = a.dot(b);
      const auto d = (ab / (a_norm * b_norm));
      if constexpr (!std::is_same_v<Jac_t, std::nullptr_t>) {
        if (Ja) *Ja = (b / (a_norm * b_norm)) - (ab * a / (a_norm * a_norm * a_norm * b_norm));
        if (Jb) *Jb = (a / (a_norm * b_norm)) - (ab * b / (a_norm * b_norm * b_norm));
      }
      return Scalar(d);
    }
  }
}

/// Compute the Mahalanobis distance between `a` and `b` with a covariance `cov`.
template <typename TA, typename TB, typename TC, typename Jac_t = std::nullptr_t,
          typename std::enable_if_t<std::is_scalar_v<TA>, int> = 0>
TA Mah(TA a, TB b, TC cov, Jac_t *Ja = nullptr, Jac_t *Jb = nullptr) {
  constexpr double eps = FloatEpsilon<TA>();
  const auto delta = (a - b) / std::sqrt(cov);
  const auto dist = std::sqrt(delta * delta);
  if constexpr (!std::is_same_v<Jac_t, std::nullptr_t>) {
    if (dist > eps) {
      if (Ja) *Ja = delta / dist;
      if (Jb) *Jb = -delta / dist;
    } else {
      if (Ja) *Ja = 0;
      if (Jb) *Jb = 0;
    }
  }
  return dist;
}

/// Compute the Mahalanobis distance between `a` and `b` with a covariance `cov`.
template <typename DA, typename DB, typename DC, typename Jac_t = std::nullptr_t>
typename DA::Scalar Mah(const MatrixBase<DA> &a, const MatrixBase<DB> &b, const MatrixBase<DC> &cov,
                        Jac_t *Ja = nullptr, Jac_t *Jb = nullptr) {
  constexpr double eps = FloatEpsilon<typename DA::Scalar>();
  using Scalar = typename DA::Scalar;
  static constexpr int Dims = DA::RowsAtCompileTime;
  using Mat = Matrix<Scalar, Dims, Dims>;
  using Vec = Vector<Scalar, Dims>;
  if constexpr (DC::ColsAtCompileTime == 1) {  //  cov is diagonal
    const Vec delta = (a - b).eval();
    const Scalar dist = std::sqrt(delta.dot(delta));
    if constexpr (!std::is_same_v<Jac_t, std::nullptr_t>) {
      if (dist > eps) {
        if (Ja) *Ja = delta / dist;
        if (Jb) *Jb = -delta / dist;
      } else {
        if (Ja) *Ja = Jac_t::Zero();
        if (Jb) *Jb = Jac_t::Zero();
      }
    }
    return dist;
  }
  if (cov.cols() == 1) {  //  cov is diagonal
    const Vec delta = (a - b).eval();
    return std::sqrt(delta.dot(delta));
  } else {
    const auto chol = Eigen::SelfAdjointView<const Mat, Upper>(cov).llt();
    const Mat L = chol.matrixL();  // L
    const Vec delta = L.template triangularView<Lower>().solve(a - b);
    const Scalar dist = std::sqrt(delta.dot(delta));
    if constexpr (!std::is_same_v<Jac_t, std::nullptr_t>) {
      constexpr double eps = 1e-8;  // TODO move out
      if (dist > eps) {
        const auto J = L.template triangularView<Lower>().solve(delta / dist).eval();
        if (Ja) *Ja = J;
        if (Jb) *Jb = -J;
      } else {
        if (Ja) *Ja = Jac_t::Zero();
        if (Jb) *Jb = Jac_t::Zero();
      }
    }
    return dist;
  }
}

/// Return scaled delta (and its jacobian J as a option) when applying a
/// Mahalanobis distance when given delta and a covariance matrix (Upper filled at least)
template <typename Derived, typename DerivedC, typename Jac_t = std::nullptr_t>
Vector<typename Derived::Scalar, Derived::RowsAtCompileTime> HalfMah(
    const MatrixBase<Derived> &delta, const MatrixBase<DerivedC> &cov, Jac_t *J = nullptr) {
  using Scalar = typename Derived::Scalar;
  static constexpr int Dims = Derived::RowsAtCompileTime;
  using Mat = Matrix<Scalar, Dims, Dims>;
  const auto chol = Eigen::SelfAdjointView<const Mat, Upper>(cov).llt();
  const Mat L = chol.matrixL();  // L
  if constexpr (!std::is_same_v<Jac_t, std::nullptr_t>)
    if (J) *J = (L.template triangularView<Lower>().solve(*J)).eval();  // J must be filled!
  return L.template triangularView<Lower>().solve(delta);
}

/// Return scaled delta (and its jacobian J as a option) when applying a
/// Mahalanobis distance when given delta and a upper triangular information matrix
template <typename Derived, typename DerivedC, typename Jac_t = std::nullptr_t>
Vector<typename Derived::Scalar, Derived::RowsAtCompileTime> HalfMahInfoU(
    const MatrixBase<Derived> &delta, const MatrixBase<DerivedC> &L, Jac_t *J = nullptr) {
  if constexpr (!std::is_same_v<Jac_t, std::nullptr_t>)
    if (J) *J = (L.template triangularView<Upper>() * (*J)).eval();
  return L.template triangularView<Upper>() * delta;
}

/// Return scaled delta (and its jacobian J as a option) when applying a
/// Mahalanobis distance when given delta and a vector of standard deviations
template <typename Derived, typename Derived2, typename Jac_t = std::nullptr_t>
auto HalfMahDiag(const MatrixBase<Derived> &delta, const MatrixBase<Derived2> &stdevs,
                 Jac_t *J = nullptr) {
  if constexpr (!std::is_same_v<Jac_t, std::nullptr_t>)
    if (J) J->noalias() = (J->array().colwise() / stdevs.array()).matrix();
  return (delta.array() / stdevs.array()).eval();
}

/// Return scaled delta (and its jacobian J as a option) when applying a scale to the delta
template <typename Derived, typename Jac_t = std::nullptr_t>
auto HalfMahIso(const MatrixBase<Derived> &delta, typename Derived::Scalar &scale,
                Jac_t *J = nullptr) {
  if constexpr (!std::is_same_v<Jac_t, std::nullptr_t>)
    if (J) *J *= scale;
  return (delta * scale).eval();
}

}  // namespace tinyopt::distances
