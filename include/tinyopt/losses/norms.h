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

#include <tinyopt/math.h>
#include <tinyopt/traits.h>
#include <type_traits>

namespace tinyopt::loss::norms {

/// Return L2 norm of a vector or scalar
template <typename T>
auto L2(const T &x) {
  constexpr bool IsMatrix = traits::is_matrix_or_array_v<T> || traits::is_sparse_matrix_v<T>;
  if constexpr (traits::is_pair_v<T>) {  // pair
    return L2(x.first, x.second);
  } else if constexpr (IsMatrix) {  // scalar
    return x.norm();
  } else {
    using std::abs;
    return abs(x);  // same as sqrt(x*x);
  }
}

/// Return L2 norm of a vector or scalar and its jacobian
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
  } else {
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
  } else {
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
    RowVector<Scalar, DimsJ> J(1, x.rows());
    if constexpr (std::is_same_v<ExportJ, bool>)
      return std::make_pair(l, RowVector<Scalar, DimsJ>(x.array().sign()));
    else
      return std::make_pair(l, (x.array().sign() * Jx_or_bool).eval());
  } else {
    return std::make_pair(l, 1);
  }
}

}  // namespace tinyopt::loss::norms
