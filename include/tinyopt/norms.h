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
// Here are common norms (L1, L2, Linf, etc.) For Mahalanobis norms, see in loss.h.
//
// Example of usage:
//
//   Vec3f x = ...;
//
//   float error = TruncatedL2(x, 0.1); // will return the truncated L2 norm
// or
//   Mat3f J = ...; // must be set to e.g identity
//   const auto [error, jac] = TruncatedL2(x, 0.1, nullptr); // will return the truncated L2 norm
//      and its jacobian
//   const auto [error, jac] = TruncatedL2(x, 0.1, &J); // will return the truncated L2 norm and its
//      jacobian using an initial jacobian 'J'

#pragma once

#include <tinyopt/math.h>
#include <tinyopt/traits.h>

namespace tinyopt::norms {

/// Return L2 norm of a vector or scalar
template <typename T>
auto L2(const T &x) {
  if constexpr (traits::is_matrix_or_array_v<T> || traits::is_sparse_matrix_v<T>) {
    return x.norm();
  } else {
    using std::abs;
    return abs(x);
  }
}

/// Return L2 norm of a vector or scalar and its jacobian
template <typename T, typename Jac_t>
auto L2(const T &x, const Jac_t *const J) {
  using Scalar = typename traits::params_trait<T>::Scalar;
  Vector<Scalar, traits::params_trait<T>::Dims> Jn;
  if constexpr (traits::is_matrix_or_array_v<T> || traits::is_sparse_matrix_v<T>) {
    static constexpr Scalar eps = 1e-7;  // TODO remove
    if (J)
      Jn = x.transpose() / (x.norm() + eps) * (*J);
    else
      Jn = x.transpose() / (x.norm() + eps);
    return std::make_pair(x.norm(), Jn);
  } else {
    using std::abs;
    Jn = (*J);
    return std::make_pair(abs(x), 1);
  }
}

/// Return L2 norm of a vector or scalar
template <typename T>
auto L2squared(const T &x) {
  if constexpr (traits::is_matrix_or_array_v<T> || traits::is_sparse_matrix_v<T>) {
    return x.squaredNorm().eval();
  } else {
    return x * x;
  }
}

/// Return L2 norm of a vector or scalar and its jacobian
template <typename T, typename Jac_t>
auto L2squared(const T &x, const Jac_t *const J) {
  if constexpr (traits::is_matrix_or_array_v<T> || traits::is_sparse_matrix_v<T>) {
    if (J)
      return std::make_pair(x.squaredNorm().eval(), (2 * x.transpose() * (*J)).eval());
    else
      return std::make_pair(x.squaredNorm().eval(), (2 * x.transpose()).eval());
  } else {
    return std::make_pair(x * x, 1);
  }
}

/// Return a Truncated L2 norm of a vector or scalar
template <typename T>
auto TruncatedL2(const T &x, double th2) {
  auto e2 = L2squared(x);
  if (e2 > th2)
    return th2;
  else
    return e2;
}

/// Return a Truncated L2 norm of a vector or scalar and its jacobian
template <typename T, typename Jac_t>
auto TruncatedL2(const T &x, double th2, const Jac_t *const J = nullptr) {
  const auto [e2, Jn] = L2squared(x, J);
  using Scalar = typename traits::params_trait<T>::Scalar;
  if (e2 > th2) {
    return std::make_pair(th2, Vector<Scalar, traits::params_trait<T>::Dims>::Zero());
  } else {
    return std::make_pair(th2, Jn);
  }
  // No changes in J here
  return e2;
}

/// Return L1 norm of a vector or scalar
template <typename T>
auto L1(const T &x) {
  if constexpr (traits::is_matrix_or_array_v<T> || traits::is_sparse_matrix_v<T>) {
    return x.template lpNorm<1>();
  } else {
    using std::abs;
    return abs(x);
  }
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
}

/// Return L infinity norm of a vector or scalar
template <typename T>
auto Linf(const T &x) {
  if constexpr (traits::is_matrix_or_array_v<T> || traits::is_sparse_matrix_v<T>) {
    return x.maxCoeff();
  } else {
    // No changes in J here
    return x;
  }
}

/// Return L infinity norm of a vector or scalar
template <typename T, typename Jac_t>
auto Linf(const T &x, const Jac_t *const J) {
  if constexpr (traits::is_matrix_or_array_v<T> || traits::is_sparse_matrix_v<T>) {
    using Scalar = typename traits::params_trait<T>::Scalar;
    Vector<Scalar, traits::params_trait<T>::Dims> Jn =
        Vector<Scalar, traits::params_trait<T>::Dims>::Zero(x.size());
    int max_idx;
    const auto max_val = x.maxCoeff(&max_idx);
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
}

}  // namespace tinyopt::norms
