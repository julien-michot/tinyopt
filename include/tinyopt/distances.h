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

#include <tinyopt/losses/mahalanobis.h>
#include <tinyopt/losses/norms.h>

namespace tinyopt::distances {

/// Compute the Euclidean L2 distance (a.k.a L2) between `a` and `b`: d(a,b) = ||a-b||
template <typename TA, typename TB, typename ExportJ = std::nullptr_t>
auto Euclidean(const TA &a, const TB &b, const ExportJ &export_jac = nullptr) {
  if constexpr (traits::is_nullptr_v<ExportJ>) {
    return losses::L2(a - b);
  } else if constexpr (traits::is_scalar_v<TA>) {
    const auto &[d, J] = losses::L2(a - b, export_jac);
    return std::make_tuple(d, J, -J);
  } else {
    const auto &[d, J] = losses::L2(a - b, export_jac);
    return std::make_tuple(d, J, (-J).eval());
  }
}

template <typename TA, typename TB, typename ExportJ = std::nullptr_t>
auto L2(const TA &a, const TB &b, const ExportJ &export_jac = nullptr) {
  return Euclidean(a, b, export_jac);
}

/// Compute the Manhattan distance (a.k.a L1) between `a` and `b`: d(a,b) = |a-b|.
template <typename TA, typename TB, typename ExportJ = std::nullptr_t>
auto Manhattan(const TA &a, const TB &b, const ExportJ &export_jac = nullptr) {
  if constexpr (traits::is_nullptr_v<ExportJ>) {
    return losses::L1(a - b);
  } else if constexpr (traits::is_scalar_v<TA>) {
    const auto &[d, J] = losses::L1(a - b, export_jac);
    return std::make_tuple(d, J, -J);
  } else {
    const auto &[d, J] = losses::L1(a - b, export_jac);
    return std::make_tuple(d, J, (-J).eval());
  }
}

template <typename TA, typename TB, typename ExportJ = std::nullptr_t>
auto L1(const TA &a, const TB &b, const ExportJ &export_jac = nullptr) {
  return Manhattan(a, b, export_jac);
}

/// Compute the L-infinity distance (a.k.a max(x)) between `a` and `b`: d(a,b) = max(a-b)
template <typename TA, typename TB, typename ExportJ = std::nullptr_t>
auto Linf(const TA &a, const TB &b, const ExportJ &export_jac = nullptr) {
  if constexpr (traits::is_nullptr_v<ExportJ>) {
    return losses::Linf(a - b);
  } else if constexpr (traits::is_scalar_v<TA>) {
    const auto &[d, J] = losses::Linf(a - b, export_jac);
    return std::make_tuple(d, J, -J);
  } else {
    const auto &[d, J] = losses::Linf(a - b, export_jac);
    return std::make_tuple(d, J, (-J).eval());
  }
}

/// Compute the cosine distance between `a` and `b`: d(a,b) = a ∠ b
template <typename TA, typename TB, typename ExportJ = std::nullptr_t>
auto Cosine(const TA &a, const TB &b, const ExportJ & = nullptr) {
  if constexpr (traits::is_nullptr_v<ExportJ>) {
    return TA(0);
  } else if constexpr (traits::is_scalar_v<TA>) {
    return std::make_tuple(TA(0), TA(0), TA(0));
  } else {  // Vectors
    using Scalar = typename TA::Scalar;
    constexpr double eps2 = FloatEpsilon2<Scalar>();
    constexpr int Dims = traits::params_trait<TA>::Dims;
    using Vec = RowVector<Scalar, Dims>;
    const auto a_norm = a.norm();
    const auto b_norm = b.norm();
    if (a_norm * b_norm < eps2) {
      return std::make_tuple(Scalar(0), Vec::Zero(a.size()), Vec::Zero(a.size()));
    } else {
      const auto ab = a.dot(b);
      const auto d = (ab / (a_norm * b_norm));
      Vec Ja = (b / (a_norm * b_norm)) - (ab * a / (a_norm * a_norm * a_norm * b_norm));
      Vec Jb = (a / (a_norm * b_norm)) - (ab * b / (a_norm * b_norm * b_norm));
      return std::make_tuple(d, Ja, Jb);
    }
  }
}

/// Compute the Mahalanobis distance between `a` and `b` with a covariance `cov`: d(a,b) = ||a-b||Σ
template <typename TA, typename TB, typename Cov_t, typename ExportJ = std::nullptr_t>
auto MahaNorm(const TA &a, const TB &b, const Cov_t &cov_or_var,
              const ExportJ &export_jac = nullptr) {
  if constexpr (traits::is_nullptr_v<ExportJ>) {
    return losses::MahaNorm(a - b, cov_or_var);
  } else if constexpr (traits::is_scalar_v<TA>) {
    const auto &[d, J] = losses::MahaNorm(a - b, cov_or_var, export_jac);
    return std::make_tuple(d, J, -J);
  } else {
    const auto &[d, J] = losses::MahaNorm(a - b, cov_or_var, export_jac);
    return std::make_tuple(d, J, (-J).eval());
  }
}

}  // namespace tinyopt::distances
