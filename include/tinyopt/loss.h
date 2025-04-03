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

/***
 * Functions defined as x' = rho(x).
 * * If `x` is a vector, a scaled vector of the same shape will be returned.
 * * If the method is passed a jacobian (J), e.g x' = rho(x, &J), then the chain rule will be
 * applied to the jacobian `J`, on the left such that and will become J = d rho(x) / d(x) * J
 */

namespace tinyopt::loss {

/**
 * @name Norms
 * @{
 */

/// Return L1 norm of a vector or scalar
template <typename T>
auto L1(const T &x) {
  if constexpr (traits::is_matrix_or_array_v<T> || traits::is_sparse_matrix_v<T>) {
    return x.template lpNorm<1>();
  } else {
    using std::abs;
    return abs(x);
  }
  /////// RETURN SUCH THAT res.t()*res == L1(res)
}

/// Return L1 norm of a vector or scalar and its jacobian
template <typename T, typename Jac_t>
auto L1(const T &x, const Jac_t *const J) {
  if constexpr (traits::is_matrix_or_array_v<T> || traits::is_sparse_matrix_v<T>) {
    using Scalar = typename traits::params_trait<T>::Scalar;
    Vector<Scalar, traits::params_trait<T>::Dims> Jn;
    if (J)
      Jn = (x.array().sign().matrix().asDiagonal() * (*J)).eval();
    else
      Jn = x.array().sign().matrix().eval();
    return std::make_pair(x.template lpNorm<1>(), Jn);
  } else {
    using std::abs;
    return std::make_pair(abs(x), 1);
  }
  /////// RETURN SUCH THAT res.t()*res == L1(res)
}

/// Return L infinity norm of a vector or scalar
template <typename T>
auto Linf(const T &x) {
  if constexpr (traits::is_matrix_or_array_v<T> || traits::is_sparse_matrix_v<T>) {
    return x.template lpNorm<Infinity>();
  } else {
    // No changes in J here
    return x;
  }
  /////// RETURN SUCH THAT res.t()*res == L1(res)
}

/// Return L infinity norm of a vector or scalar
template <typename T, typename Jac_t>
auto Linf(const T &x, const Jac_t *const J) {
  if constexpr (traits::is_matrix_or_array_v<T> || traits::is_sparse_matrix_v<T>) {
    using Scalar = typename traits::params_trait<T>::Scalar;
    Vector<Scalar, traits::params_trait<T>::Dims> Jn =
        Vector<Scalar, traits::params_trait<T>::Dims>::Zero(x.size());
    int max_idx;
    const auto max_val = x.cwiseAbs().maxCoeff(&max_idx);
    if (J) {
      Jn[max_idx] = 1;
      Jn = (Jn.transpose() * (*J)).transpose().eval();  // TODO speed this up & check...
    } else {
      Jn[max_idx] = 1;
    }
    return std::make_pair(max_val, Jn);
  } else {
    return std::make_pair(x, 1);
  }
  /////// RETURN SUCH THAT res.t()*res == Linf(res)
}

/// Return scaled delta (and its jacobian J as a option) when applying a
/// Mahalanobis distance when given delta and a covariance matrix (Upper filled at least)
template <typename Derived, typename DerivedC, typename Jac_t = std::nullptr_t>
Vector<typename Derived::Scalar, Derived::RowsAtCompileTime> Manhattan(
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
Vector<typename Derived::Scalar, Derived::RowsAtCompileTime> ManhattanInfoU(
    const MatrixBase<Derived> &delta, const MatrixBase<DerivedC> &L, Jac_t *J = nullptr) {
  if constexpr (!std::is_same_v<Jac_t, std::nullptr_t>)
    if (J) *J = (L.template triangularView<Upper>() * (*J)).eval();
  return L.template triangularView<Upper>() * delta;
}

/// Return scaled delta (and its jacobian J as a option) when applying a
/// Mahalanobis distance when given delta and a vector of standard deviations
template <typename Derived, typename Derived2, typename Jac_t = std::nullptr_t>
auto ManhattanDiag(const MatrixBase<Derived> &delta, const MatrixBase<Derived2> &stdevs,
                   Jac_t *J = nullptr) {
  if constexpr (!std::is_same_v<Jac_t, std::nullptr_t>)
    if (J) J->noalias() = (J->array().colwise() / stdevs.array()).matrix();
  return (delta.array() / stdevs.array()).eval();
}

/// Return scaled delta (and its jacobian J as a option) when applying a scale to the delta
template <typename Derived, typename Jac_t = std::nullptr_t>
auto Iso(const MatrixBase<Derived> &delta, typename Derived::Scalar &scale, Jac_t *J = nullptr) {
  if constexpr (!std::is_same_v<Jac_t, std::nullptr_t>)
    if (J) *J *= scale;
  return (delta * scale).eval();
}

/**
 * @name M-Estimators - Robust 'Norms'
 * @{
 */

/// Update
template <typename T, typename Jac_t = std::nullptr_t>
auto TruncatedL2(const T &res, typename traits::params_trait<T>::Scalar th2, Jac_t *J = nullptr) {
  const auto e2 = res.squaredNorm();
  if (e2 > th2) {
    if constexpr (!std::is_same_v<Jac_t, std::nullptr_t>)
      if (J) J->setZero();
  }
  return res;  // no scaling
  /////// RETURN SUCH THAT res.t()*res ==th2 so res *= (th2 / e2)????
}

/// Return a Truncated L2 norm of a vector or scalar
template <typename T>
auto TruncatedL2(const T &x, typename traits::params_trait<T>::Scalar th2) {
  auto e2 = SquaredNorm(x);
  if (e2 > th2)
    return th2;
  else
    return e2;
  /////// RETURN SUCH THAT res.t()*res ==th2 so res *= (th2 / e2)????
}

/// Return a Truncated L2 norm of a vector or scalar and its jacobian
template <typename T, typename Jac_t>
auto TruncatedL2(const T &x, typename traits::params_trait<T>::Scalar th2,
                 const Jac_t *const J = nullptr) {
  const auto [e2, Jn] = SquaredNorm(x, J);
  using Scalar = typename traits::params_trait<T>::Scalar;
  if (e2 > th2) {
    return std::make_pair(th2, Vector<Scalar, traits::params_trait<T>::Dims>::Zero());
  } else {
    return std::make_pair(th2, Jn);
  }
  // No changes in J here
  return e2;
  /////// RETURN SUCH THAT res.t()*res ==th2 so res *= (th2 / e2)????
}

// TODO Arctan, Huber, Tukey, Cauchy, GemanMcClure, lakeZisserman, Welsch, Fair,
// https://arxiv.org/pdf/1701.03077.pdf

/** @} */

/**
 * @name Activation functions
 * @{
 */

// TODO Sigmoid = 1/(1+e^-x),           derivative = Sigmoid(x) * (1 - Sigmoid(x))
// TODO Tanh = (e^x-e^-x)/(e^x+e^-x),   derivative = 2 - Tanh(x)^2
// TODO ReLU = max(0, x),               derivative = {x>0:1, x<=0:0}
// TODO LeakyReLU = {x>0:x, x<=0:a*x},  derivative = {x>0:1, x<=0:a}
// TODO Softmax = e^xi / sum(e^x),      jacobian = {i=j: si(x)*(1-si(x)) , i!=j: -si(x)*sj(x)}

/** @} */

}  // namespace tinyopt::loss
