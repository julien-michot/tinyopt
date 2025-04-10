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

namespace tinyopt::losses {

/**
 * @name Mahalanobis Distances
 * @{
 */

/// Compute the Squared Mahalanobis distance of ´x´ with a covariance `cov`: n(x) = ||x||Σ²
template <typename T, typename Cov_t, typename ExportJ = std::nullptr_t>
auto SquaredMahaNorm(const T &x, const Cov_t &cov_or_var, const ExportJ &Jx_or_bool = nullptr,
                     bool add_scale = true) {
  using Scalar = typename traits::params_trait<T>::Scalar;
  constexpr int Dims = traits::params_trait<T>::Dims;
  if constexpr (traits::is_scalar_v<T>) {  // scalar
    const T s = cov_or_var < FloatEpsilon<T>() ? T(1.0) : T(T(1.0) / cov_or_var);
    const T n2 = x * x * s;  // same as sqrt(x²*s)
    if constexpr (traits::is_bool_v<ExportJ>) {
      const T J = add_scale ? Scalar(2) * s * x : s * x;
      return std::make_pair(n2, J);
    } else if constexpr (!traits::is_nullptr_v<ExportJ>) {
      const T J = add_scale ? Scalar(2) * s * x : s * x;
      return std::make_pair(n2, T(J * Jx_or_bool));
    } else {
      return n2;
    }
  } else if constexpr (traits::is_nullptr_v<ExportJ>) {  // x is a vector, no jacobian export
    const auto cov_vars = cov_or_var.template cast<Scalar>().eval();
    if (x.cols() != 1) throw std::invalid_argument("'x' must be a vector");
    Scalar n2 = Scalar(0.0);
    if constexpr (traits::params_trait<Cov_t>::ColsAtCompileTime == 1) {  // variances
      n2 = x.dot(cov_vars.cwiseInverse().asDiagonal() * x);
    } else {  // covariance matrix
      if (cov_vars.cols() > 1) {
        const auto I = InvCov(cov_vars);
        if (!I.has_value())
          throw std::invalid_argument("Covariance is not invertible, make sure it is");
        n2 = x.dot(I.value() * x);
      } else if constexpr (traits::params_trait<Cov_t>::ColsAtCompileTime ==
                           Dynamic) {  // variances
        n2 = x.dot(cov_vars.cwiseInverse().asDiagonal() * x);
      }
    }
    return n2;
  } else {  // x is a vector, exporting jacobian
    if (x.cols() != 1) throw std::invalid_argument("'x' must be a vector");
    const auto cov_vars = cov_or_var.template cast<Scalar>().eval();
    Scalar n2 = Scalar(0.0);
    Vector<Scalar, Dims> Jt(x.size());

    if constexpr (traits::params_trait<Cov_t>::ColsAtCompileTime == 1) {  // variances
      Jt = cov_vars.cwiseInverse().asDiagonal() * x;
      n2 = x.dot(Jt);
    } else {  // covariance matrix
      if (cov_vars.cols() > 1) {
        const auto I = InvCov(cov_vars);
        if (!I.has_value())
          throw std::invalid_argument("Covariance is not invertible, make sure it is");
        Jt = I.value() * x;
        n2 = x.dot(Jt);
      } else if constexpr (traits::params_trait<Cov_t>::ColsAtCompileTime ==
                           Dynamic) {  // variances
        Jt = cov_vars.cwiseInverse().asDiagonal() * x;
        n2 = x.dot(Jt);
      }
    }
    if (add_scale) Jt *= Scalar(2);
    if constexpr (traits::is_bool_v<ExportJ>)
      return std::make_pair(n2, Jt.transpose().eval());
    else
      return std::make_pair(n2, (Jt.transpose() * Jx_or_bool).eval());
  }
}
/// Compute the Mahalanobis distance of ´x´ with a covariance `cov`: n(x) = ||x||Σ
template <typename T, typename Cov_t, typename ExportJ = std::nullptr_t>
auto MahaNorm(const T &x, const Cov_t &cov_or_var, const ExportJ &Jx_or_bool = nullptr) {
  using Scalar = typename traits::params_trait<T>::Scalar;
  using std::sqrt;
  constexpr bool add_scale = false;
  if constexpr (traits::is_nullptr_v<ExportJ>) {
    const auto &n2 = SquaredMahaNorm(x, cov_or_var, nullptr, add_scale);
    return sqrt(n2);
  } else if constexpr (traits::is_scalar_v<T>) {  // scalar
    using std::abs;
    const auto &[n2, J] = SquaredMahaNorm(x, cov_or_var, Jx_or_bool, add_scale);
    const auto n = sqrt(n2);
    const auto s = n > FloatEpsilon<Scalar>() ? n : Scalar(1);
    return std::make_pair(n, J / s);
  } else {
    const auto &[n2, J] = SquaredMahaNorm(x, cov_or_var, Jx_or_bool, add_scale);
    const auto n = sqrt(n2);
    const auto s = n > FloatEpsilon<Scalar>() ? n : Scalar(1);
    return std::make_pair(n, (J / s).eval());
  }
}

/// Return the Whitened/Sphered/Decorrelated residuals (and its jacobian as a option) when
/// applying a Mahalanobis norm when given residuals and a covariance matrix (Upper filled at least)
template <typename Derived, typename Cov_t, typename ExportJ = std::nullptr_t>
auto MahaWhitened(const MatrixBase<Derived> &res, const Cov_t &cov_stevs,
                  const ExportJ &Jx_or_bool = nullptr) {
  using Scalar = typename Derived::Scalar;
  if constexpr (traits::is_scalar_v<Cov_t>) {  // iso
    const auto res2 = (res / cov_stevs).eval();
    // Jacobian
    if constexpr (traits::is_nullptr_v<ExportJ>)
      return res2;
    else if constexpr (traits::is_bool_v<ExportJ>)
      return std::make_pair(res2, 1.0 / cov_stevs);
    else
      return std::make_pair(res2, (Jx_or_bool / cov_stevs).eval());

  } else if constexpr (traits::params_trait<Cov_t>::ColsAtCompileTime ==
                       1) {  // standard deviations
    const auto res2 = (res.array() / cov_stevs.array()).matrix().eval();
    using Mat = Matrix<Scalar, Derived::RowsAtCompileTime, Derived::RowsAtCompileTime>;

    // Jacobian
    if constexpr (traits::is_nullptr_v<ExportJ>)
      return res2;
    else if constexpr (traits::is_bool_v<ExportJ>)
      return std::make_pair(res2, Mat(cov_stevs.cwiseInverse().asDiagonal()));
    else
      return std::make_pair(res2, (cov_stevs.cwiseInverse().asDiagonal() * Jx_or_bool).eval());

  } else {
    const auto cov = cov_stevs.template cast<Scalar>().eval();
    {  // cov matrix
      static constexpr int Dims = Derived::RowsAtCompileTime;
      using Mat = Matrix<Scalar, Dims, Dims>;
      const auto chol = Eigen::SelfAdjointView<const Mat, Upper>(cov).llt();
      const auto res2 = chol.matrixL().solve(res).eval();

      // Jacobian
      if constexpr (traits::is_nullptr_v<ExportJ>)
        return res2;
      else if constexpr (traits::is_bool_v<ExportJ>)
        return std::make_pair(res2, (chol.matrixL().solve(Mat::Identity())).eval());
      else
        return std::make_pair(res2, (chol.matrixL().solve(Jx_or_bool)).eval());

    }  // TODO if constexpr (traits::params_trait<Cov_t>::ColsAtCompileTime == Dynamic)
  }
}

/// Return the Whitened/Sphered/Decorrelated residuals (and its jacobian as a option) when
/// applying a Mahalanobis norm when given residuals and a Upper Information matrix
template <typename Derived, typename DerivedC, typename ExportJ = std::nullptr_t>
auto MahaWhitenedInfoU(const MatrixBase<Derived> &res, const MatrixBase<DerivedC> &U,
                       const ExportJ &Jx_or_bool = nullptr) {
  auto UU = U.template triangularView<Upper>();
  const auto res2 = (UU * res).eval();
  // Jacobian
  if constexpr (traits::is_nullptr_v<ExportJ>)
    return res2;
  else if constexpr (traits::is_bool_v<ExportJ>)
    return std::make_pair(res2, U.eval());
  else
    return std::make_pair(res2, (UU * Jx_or_bool).eval());
}

/** @} */

}  // namespace tinyopt::losses
