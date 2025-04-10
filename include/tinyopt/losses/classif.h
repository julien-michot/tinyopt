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
 * @name Classification losses
 * @{
 */
/// @brief Softmax = e^xi / sum(e^x),  jacobian = {i=j: si(x)*(1-si(x)) , i!=j: -si(x)*sj(x)}
template <typename T, typename ExportJ = std::nullptr_t>
auto Softmax(const T &x, const ExportJ &Jx_or_bool = nullptr) {
  constexpr bool IsMatrix = traits::is_matrix_or_array_v<T> || traits::is_sparse_matrix_v<T>;
  if constexpr (traits::is_pair_v<T>) {  // pair
    return Softmax(x.first, x.second);
  } else if constexpr (!IsMatrix) {  // scalar
    if constexpr (std::is_null_pointer_v<ExportJ>)
      return 1;
    else if constexpr (traits::is_matrix_or_array_v<ExportJ>)
      return std::make_pair(1, Jx_or_bool);
    else
      return std::make_pair(1, 1);
  } else {  // Matrix
    using std::exp;
    using Scalar = typename T::Scalar;
    constexpr Index Dims = T::RowsAtCompileTime;
    const auto si = x.array().exp().matrix().eval();
    const Scalar sum = si.sum();
    const T out = x.unaryExpr([sum](Scalar v) { return exp(v) / sum; }).eval();
    if constexpr (std::is_null_pointer_v<ExportJ>) {
      return out;
    } else {
      Matrix<Scalar, Dims, Dims> Jo(x.rows(), x.rows());
      Jo.setZero();
      for (int c = 0; c < Jo.cols(); ++c) {
        for (int r = c; r < Jo.rows(); ++r) {
          Jo(r, c) = r == c ? out[c] * (1.0 - out[c]) : -out[r] * out[c];
        }
      }
      Jo.template triangularView<Upper>() = Jo.template triangularView<Lower>().transpose();
      return std::make_pair(out, (Jo * Jx_or_bool).eval());
    }
  }
}

/// @brief Safe Softmax = e^(xi-max(xi)) / sum(e^(x-max(xi))
template <typename T, typename ExportJ = std::nullptr_t>
auto SafeSoftmax(const T &x, const ExportJ &Jx_or_bool = nullptr) {
  constexpr bool IsMatrix = traits::is_matrix_or_array_v<T> || traits::is_sparse_matrix_v<T>;
  if constexpr (traits::is_pair_v<T>) {  // pair
    return SafeSoftmax(x.first, x.second);
  } else if constexpr (!IsMatrix) {  // scalar
    if constexpr (std::is_null_pointer_v<ExportJ>)
      return std::make_pair(1, 1);
    else if constexpr (traits::is_matrix_or_array_v<ExportJ>)
      return std::make_pair(1, Jx_or_bool);
    else
      return 1;
  } else {  // Matrix
    using std::exp;
    using Scalar = typename T::Scalar;
    constexpr Index Dims = T::RowsAtCompileTime;
    const Scalar mx = x.maxCoeff();
    const auto si = (x.array() - mx).exp().matrix().eval();
    const Scalar sum = si.sum();
    const T out = x.unaryExpr([mx, sum](Scalar v) { return exp(v - mx) / sum; });
    if constexpr (std::is_null_pointer_v<ExportJ>) {
      return out;
    } else {
      Matrix<Scalar, Dims, Dims> Jo(x.rows(), x.rows());
      Jo.setZero();
      for (int c = 0; c < Jo.cols(); ++c) {
        for (int r = c; r < Jo.rows(); ++r) {
          Jo(r, c) = r == c ? out[c] * (1.0 - out[c]) : -out[r] * out[c];
        }
      }
      Jo.template triangularView<Upper>() = Jo.template triangularView<Lower>().transpose();
      return std::make_pair(out, (Jo * Jx_or_bool).eval());
    }
  }
}

/** @} */
}  // namespace tinyopt::losses
