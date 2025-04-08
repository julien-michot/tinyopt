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

#include <tinyopt/losses/norms.h>

namespace tinyopt::losses {

/**
 * @name M-Estimators and Robust Norms
 * @{
 */

// TODO Arctan, Huber, Tukey, Cauchy, GemanMcClure, lakeZisserman, Welsch, Fair,
// https://arxiv.org/pdf/1701.03077.pdf

/// Return a Hard Truncated L2 norm of a vector or scalar
template <typename T, typename ParamType = float>
auto TruncatedL2(const T &x, ParamType th) {
  const auto l = L2(x);
  return l <= th ? l : decltype(l)(th);
}

/// Return a Hard Truncated L2 norm of a vector or scalar
template <typename T, typename ParamType, typename ExportJ>
auto TruncatedL2(const T &x, ParamType th, const ExportJ &Jx_or_bool) {
  constexpr bool IsMatrix = traits::is_matrix_or_array_v<T> || traits::is_sparse_matrix_v<T>;
  const auto &out = L2(x, Jx_or_bool);
  const auto &l = out.first;
  if (l > th) {                // outlier
    if constexpr (IsMatrix) {  // matrix
      using Scalar = typename traits::params_trait<T>::Scalar;
      constexpr int DimsJ = traits::params_trait<T>::Dims;
      RowVector<Scalar, DimsJ> J = RowVector<Scalar, DimsJ>::Zero(x.rows());
      if constexpr (std::is_same_v<ExportJ, bool>)
        return std::make_pair(th, J);
      else
        return std::make_pair(th, (J * Jx_or_bool).eval());
    } else {  // scalar
      using Scalar = typename traits::params_trait<T>::Scalar;
      return std::make_pair(th, Scalar(0));
    }
  } else {  // inlier
    return out;
  }
}

/** @} */

}  // namespace tinyopt::losses
