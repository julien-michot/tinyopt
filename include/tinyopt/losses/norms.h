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
auto SquaredL2(const T &x, const ExportJ &Jx_or_bool, bool add_scale = true) {
  constexpr bool IsMatrix = traits::is_matrix_or_array_v<T> || traits::is_sparse_matrix_v<T>;
  const auto l = SquaredL2(x);
  if constexpr (IsMatrix) {
    using Scalar = typename traits::params_trait<T>::Scalar;
    RowVector<Scalar, traits::params_trait<T>::Dims> J = x.transpose();
    if (add_scale) J *= Scalar(2.0);
    if constexpr (traits::is_bool_v<ExportJ>)
      return std::make_pair(l, J);
    else
      return std::make_pair(l, (J * Jx_or_bool).eval());
  } else {  // scalar
    return std::make_pair(l, add_scale ? T(2.0) * x : x);
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
    if constexpr (traits::is_bool_v<ExportJ>)
      return std::make_pair(l, J.transpose().eval());
    else
      return std::make_pair(l, (J.transpose() * Jx_or_bool).eval());
  } else {  // scalar
    return std::make_pair(l, 1);
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
    if constexpr (traits::is_bool_v<ExportJ>)
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
    if constexpr (traits::is_bool_v<ExportJ>)
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

}  // namespace tinyopt::losses
