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
#include <cstddef>
#include <type_traits>
#include <utility>

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
Vector<typename Derived::Scalar, Derived::RowsAtCompileTime> Maha(const MatrixBase<Derived> &delta,
                                                                  const MatrixBase<DerivedC> &cov,
                                                                  Jac_t *J = nullptr) {
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
Vector<typename Derived::Scalar, Derived::RowsAtCompileTime> MahaInfoU(
    const MatrixBase<Derived> &delta, const MatrixBase<DerivedC> &L, Jac_t *J = nullptr) {
  if constexpr (!std::is_same_v<Jac_t, std::nullptr_t>)
    if (J) *J = (L.template triangularView<Upper>() * (*J)).eval();
  return L.template triangularView<Upper>() * delta;
}

/// Return scaled delta (and its jacobian J as a option) when applying a
/// Mahalanobis distance when given delta and a vector of standard deviations
template <typename Derived, typename Derived2, typename Jac_t = std::nullptr_t>
auto MahaDiag(const MatrixBase<Derived> &delta, const MatrixBase<Derived2> &stdevs,
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

/**
Usage:

res2 = Huber(res)
res2, J2 = Huber(res, J)

Linf, Softmax change

Typical use:
  NLLS:
    res2, J2 = Huber(res, J) or res2, J2 = Sigmoid(res)
    error = L2(res2); // or L2² to save one std::sqrt
    H += Jt*J;
    g += Jt*res2
  GENERAL:
    error, J = L2(Huber(res2, J)); or Sigmoid(L2(Huber(res2, J)))
    g += Jt*res2


template <typename T, typename Jac_t>
auto Huber(const T &x, typename traits::params_trait<T>::Scalar th,
           const Jac_t *const J = nullptr) {
  const auto e = x.norm();
  using Scalar = typename traits::params_trait<T>::Scalar;
  if (e > th) {
    return 2 * th * e - th2;
    if (jac) (*jac) *= std::max(std::numeric_limits<double>::min(), th / e);
  } else {
    return std::make_pair(th2, Jn);
  }
  /////// RETURN SUCH THAT res.t()*res ==th2 so res *= (th2 / e2)????
}
*/

/** @} */

/**
 * @name Activation functions (aj = rho(zj), with e.g. zj = Σwij*xi+bj)
 * @{
 */

/// @brief Sigmoid = 1/(1+e^-x), derivative = Sigmoid(x) * (1 - Sigmoid(x))
template <typename T, typename Jac_t = std::nullptr_t>
auto Sigmoid(const T &x, const Jac_t &J = nullptr) {
  using std::exp;
  if constexpr (std::is_scalar_v<T>) {  // scalar
    const auto o = T(1.0) / (T(1.0) + exp(-x));
    if constexpr (std::is_null_pointer_v<Jac_t>)
      return o;
    else if constexpr (traits::is_matrix_or_array_v<Jac_t>)
      return std::make_pair(o, J * (o * (1.0 - o)));
    else
      return std::make_pair(o, o * (1.0 - o));
  } else {  // Matrix -> per element relu
    using Scalar = typename T::Scalar;
    const T out =
        x.unaryExpr([](Scalar v) { return Scalar(1.0) / (Scalar(1.0) + exp(-v)); }).eval();
    if constexpr (std::is_null_pointer_v<Jac_t>) {
      return out;
    } else {
      auto Jo = out.unaryExpr([](Scalar v) { return v * (Scalar(1.0) - v); });
      return std::make_pair(out, (Jo.transpose() * J.transpose()).eval());
    }
  }
}

/// @brief Tanh = (e^x-e^-x)/(e^x+e^-x),   derivative = 2 - Tanh(x)^2
template <typename T, typename Jac_t = std::nullptr_t>
auto Tanh(const T &x, const Jac_t &J = nullptr) {
  using std::exp;
  if constexpr (std::is_scalar_v<T>) {  // scalar
    const auto o = (exp(x) - exp(-x)) / (exp(x) + exp(-x));
    if constexpr (std::is_null_pointer_v<Jac_t>)
      return o;
    else if constexpr (traits::is_matrix_or_array_v<Jac_t>)
      return std::make_pair(o, J * (2 - o * o));
    else
      return std::make_pair(o, 2 - o * o);
  } else {  // Matrix -> per element relu
    using Scalar = typename T::Scalar;
    const T out =
        x.unaryExpr([](Scalar v) { return (exp(v) - exp(-v)) / (exp(v) + exp(-v)); }).eval();
    if constexpr (std::is_null_pointer_v<Jac_t>) {
      return out;
    } else {
      auto Jo = out.unaryExpr([](Scalar v) { return 2 - v * v; });
      return std::make_pair(out, (Jo.transpose() * J).eval());
    }
  }
}

/// @brief ReLU = max(0, x),  derivative = {x>0:1, x<=0:0}
template <typename T, typename Jac_t = std::nullptr_t>
auto ReLU(const T &x, const Jac_t &J = nullptr) {
  if constexpr (std::is_scalar_v<T>) {  // scalar
    const auto o = x > 0 ? x : 0;
    if constexpr (std::is_null_pointer_v<Jac_t>)
      return o;
    else if constexpr (traits::is_matrix_or_array_v<Jac_t>)
      return std::make_pair(o, J * (x > 0 ? 1 : 0));
    else
      return std::make_pair(o, x > 0 ? 1 : 0);
  } else {  // Matrix -> per element relu
    using Scalar = typename T::Scalar;
    const T out = x.unaryExpr([](Scalar v) { return v > 0 ? v : 0; }).eval();
    if constexpr (std::is_null_pointer_v<Jac_t>) {
      return out;
    } else {
      auto Jo = x.unaryExpr([](Scalar v) { return v > 0 ? 1 : 0; });
      return std::make_pair(out, (Jo.transpose() * J).eval());
    }
  }
}

/// @brief LeakyReLU = {x>0:x, x<=0:a*x}, derivative = {x>0:1, x<=0:a}
template <typename T, typename Jac_t = std::nullptr_t>
auto LeakyReLU(const T &x, double a = 0.01, const Jac_t &J = nullptr) {
  if constexpr (std::is_scalar_v<T>) {  // scalar
    const auto o = x > 0 ? x : a * x;
    if constexpr (std::is_null_pointer_v<Jac_t>)
      return o;
    else if constexpr (traits::is_matrix_or_array_v<Jac_t>)
      return std::make_pair(o, J * (x > 0 ? 1 : a));
    else
      return std::make_pair(o, x > 0 ? 1 : a);
  } else {  // Matrix -> per element leaky relu
    using Scalar = typename T::Scalar;
    const T out = x.unaryExpr([a](Scalar v) { return v > 0 ? v : a * v; }).eval();
    if constexpr (std::is_null_pointer_v<Jac_t>) {
      return out;
    } else {
      auto Jo = x.unaryExpr([a](Scalar v) { return v > 0 ? 1 : a; });
      return std::make_pair(out, (Jo.transpose() * J).eval());
    }
  }
}

/// @brief Softmax = e^xi / sum(e^x),  jacobian = {i=j: si(x)*(1-si(x)) , i!=j: -si(x)*sj(x)}
template <typename T, typename Jac_t = std::nullptr_t>
auto Softmax(const T &x, const Jac_t &J = nullptr) {
  if constexpr (std::is_scalar_v<T>) {  // scalar
    if constexpr (std::is_null_pointer_v<Jac_t>)
      return 1;
    else if constexpr (traits::is_matrix_or_array_v<Jac_t>)
      return std::make_pair(1, J);
    else
      return std::make_pair(1, 1);
  } else {  // Matrix
    using std::exp;
    using Scalar = typename T::Scalar;
    constexpr int Dims = T::RowsAtCompileTime;
    const auto si = x.exp().eval();
    const Scalar sum = si.sum();
    const T out = x.unaryExpr([sum](Scalar v) { return exp(v) / sum; }).eval();
    if constexpr (std::is_null_pointer_v<Jac_t>) {
      return out;
    } else {
      Matrix<Scalar, Dims, Dims> Jo(x.rows(), x.rows());
      for (int c = 0; c < Jo.cols(); ++c) {
        for (int r = c; r < Jo.rows(); ++r) {
          Jo(r, c) = r == c ? si[c] * (1.0 - si[c]) : -si[r] * si[c];
        }
      }
      Jo.template triangularView<Lower>() = Jo.template triangularView<Upper>();
      return std::make_pair(out, (Jo * J).eval());
    }
  }
}

/// @brief Safe Softmax = e^(xi-max(xi)) / sum(e^(x-max(xi))
template <typename T, typename Jac_t = std::nullptr_t>
auto SafeSoftmax(const T &x, const Jac_t &J = nullptr) {
  if constexpr (std::is_scalar_v<T>) {  // scalar
    if constexpr (std::is_null_pointer_v<Jac_t>)
      return std::make_pair(1, 1);
    else if constexpr (traits::is_matrix_or_array_v<Jac_t>)
      return std::make_pair(1, J);
    else
      return 1;
  } else {  // Matrix
    using std::exp;
    using Scalar = typename T::Scalar;
    constexpr int Dims = T::RowsAtCompileTime;
    const Scalar mx = x.maxCoeff();
    const auto si = (x.array() - mx).exp().eval();
    const Scalar sum = si.sum();
    const T out = x.unaryExpr([mx, sum](Scalar v) { return exp(v - mx) / sum; });
    if constexpr (std::is_null_pointer_v<Jac_t>) {
      return out;
    } else {
      Matrix<Scalar, Dims, Dims> Jo(x.rows(), x.rows());
      for (int c = 0; c < Jo.cols(); ++c) {
        for (int r = c; r < Jo.rows(); ++r) {
          Jo(r, c) = r == c ? si[c] * (1.0 - si[c]) : -si[r] * si[c];
        }
      }
      Jo.template triangularView<Lower>() = Jo.template triangularView<Upper>();
      return std::make_pair(out, (Jo * J).eval());
    }
  }
}

/** @} */

}  // namespace tinyopt::loss
