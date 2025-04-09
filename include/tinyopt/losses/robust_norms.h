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

/// If you land here, put your shades one because those fat functions are still hurting my eyes as
/// of today.

#pragma once

#include <tinyopt/math.h>
#include <tinyopt/traits.h>

#include <tinyopt/losses/norms.h>

namespace tinyopt::losses {

/**
 * @name M-Estimators and Robust Norms
 * @brief These functions return a norm and a scale to apply on the jacobian or the scaled jacobian
 * @{
 */

///////////////////// TruncatedL2 ////////////////////////

/// @brief Hard Truncation: Clip the given norm `n` to a max of `th`,
/// also return the scale or scaled jacobian if given as input `Jx_or_bool`
template <typename T, typename ExportJ = std::nullptr_t>
auto Truncated(const T &n, typename traits::params_trait<T>::Scalar th,
               const ExportJ &Jx_or_bool = nullptr) {
  if constexpr (traits::is_pair_v<T>) {  // pair (norm, jacobian)
    return Truncated(n.first, th, n.second);
  } else {
    static_assert(!traits::is_matrix_or_array_v<T>, "`n` must be a scalar");
    const auto l = n <= th ? n : T(th);
    if constexpr (std::is_null_pointer_v<ExportJ>) {
      return l;
    } else if constexpr (std::is_same_v<ExportJ, bool>) {
      return std::make_pair(l, n <= th ? T(1) : T(0));
    } else {  // Jx_or_bool is Jx
      assert(Jx_or_bool.rows() == 1);
      if (n <= th) {
        return std::make_pair(l, Jx_or_bool);
      } else {
        return std::make_pair(l, ExportJ::Zero(1, Jx_or_bool.cols()).eval());
      }
    }
  }
}

/// @brief Hard Truncated L2 Norm: Clip the L2 norm of `x` to a max of `th`,
/// also return the scale or scaled jacobian if given as input `Jx_or_bool`
template <typename T, typename ExportJ = std::nullptr_t>
auto TruncatedNorm(const T &x, typename traits::params_trait<T>::Scalar th,
                   const ExportJ &Jx_or_bool = nullptr) {
  return Truncated(L2(x, Jx_or_bool), th);
}

///////////////////// Huber ////////////////////////

/// @brief Huber: Return a scaled norm @f$ n'= n if n < th, else n'= \sqrt(2.0 * th * n - th²) @f$,
/// also return the scale or scaled jacobian if given as input `Jx_or_bool`
template <typename T, typename ExportJ = std::nullptr_t>
auto Huber(const T &n, typename traits::params_trait<T>::Scalar th,
           const ExportJ &Jx_or_bool = nullptr) {
  if constexpr (traits::is_pair_v<T>) {  // pair (norm, jacobian)
    return Huber(n.first, th, n.second);
  } else {
    using std::max;
    using std::min;
    using std::sqrt;
    static_assert(!traits::is_matrix_or_array_v<T>, "`n` must be a scalar");

    if (n <= th) {  // Inlier
      const auto l = n;
      if constexpr (std::is_null_pointer_v<ExportJ>) {  // no jacobians
        return l;
      } else {
        if constexpr (std::is_same_v<ExportJ, bool>)
          return std::make_pair(l, T(1));
        else
          return std::make_pair(l, Jx_or_bool);
      }
    } else {  // Outlier
      const auto l = sqrt(T(2.0) * th * n - th * th);
      if constexpr (std::is_null_pointer_v<ExportJ>) {  // no jacobians
        return l;
      } else {
        const T J_scale = max<T>(std::numeric_limits<T>::min(), th / n);
        if constexpr (std::is_same_v<ExportJ, bool>)
          return std::make_pair(l, J_scale);
        else
          return std::make_pair(l, (J_scale * Jx_or_bool).eval());
      }
    }
  }
}

/// @brief Huber: Return a scaled norm of 'x' such that @f$ n'= ||x|| if ||x|| < th, else n'= 2.0 *
/// th * n - th² @f$, also return the scale or scaled jacobian if given as input `Jx_or_bool`
template <typename T, typename ExportJ = std::nullptr_t>
auto HuberNorm(const T &x, typename traits::params_trait<T>::Scalar th,
               const ExportJ &Jx_or_bool = nullptr) {
  return Huber(L2(x, Jx_or_bool), th);
}

///////////////////// Tuckey ////////////////////////

/// @brief Tukey: Return a scaled norm @f$ n'= n if n < th, else n'= 2.0 * th * n - th² @f$,
/// also return the scale or scaled jacobian if given as input `Jx_or_bool`
template <typename T, typename ExportJ = std::nullptr_t>
auto Tukey(const T &n, typename traits::params_trait<T>::Scalar th,
           const ExportJ &Jx_or_bool = nullptr) {
  // TODO better use n² and th² as input
  if constexpr (traits::is_pair_v<T>) {  // pair (norm, jacobian)
    return Tukey(n.first, th, n.second);
  } else {
    static_assert(!traits::is_matrix_or_array_v<T>, "`n` must be a scalar");
    if (n <= th) {  // Inlier
      const auto n2 = n * n, th2 = th * th;
      const auto s = T(1.0) - n2 / th2, s2 = s * s;
      const auto l = th * sqrt(T(1.0) - s2 * s);
      if constexpr (std::is_null_pointer_v<ExportJ>) {  // no jacobians
        return l;
      } else {
        const auto J_scale = T(3.0) * (th2 - n2) * (th2 - n2) / (th2 * th2);
        if constexpr (std::is_same_v<ExportJ, bool>)
          return std::make_pair(l, J_scale);
        else
          return std::make_pair(l, (J_scale * Jx_or_bool).eval());
      }
    } else {  // Outlier
      const T l = T(th);
      if constexpr (std::is_null_pointer_v<ExportJ>) {  // no jacobians
        return l;
      } else {
        if constexpr (std::is_same_v<ExportJ, bool>)
          return std::make_pair(l, T(0));
        else
          return std::make_pair(l, ExportJ::Zero(1, Jx_or_bool.cols()).eval());
      }
    }
  }
}

/// @brief Tukey: Return a scaled norm of 'x' such that @f$ n'= ||x|| if ||x|| < th, else n'= 2.0 *
/// th * n - th² @f$, also return the scale or scaled jacobian if given as input `Jx_or_bool`
template <typename T, typename ExportJ = std::nullptr_t>
auto TukeyNorm(const T &x, typename traits::params_trait<T>::Scalar th,
               const ExportJ &Jx_or_bool = nullptr) {
  return Tukey(L2(x, Jx_or_bool), th);
}

///////////////////// Arctan ////////////////////////

/// @brief Arctan: Return a scaled norm
/// @f$ n' = \sqrt(th * \atan(n² / th)) @f$,
/// also return the scale or scaled jacobian if given as input `Jx_or_bool`
template <typename T, typename ExportJ = std::nullptr_t>
auto Arctan(const T &n, typename traits::params_trait<T>::Scalar th,
            const ExportJ &Jx_or_bool = nullptr) {
  if constexpr (traits::is_pair_v<T>) {  // pair (norm, jacobian)
    return Arctan(n.first, th, n.second);
  } else {
    using std::atan2;
    using std::max;
    using std::sqrt;
    static_assert(!traits::is_matrix_or_array_v<T>, "`n` must be a scalar");
    const auto l = sqrt(th * atan2(n * n, T(th)));
    if constexpr (std::is_null_pointer_v<ExportJ>) {  // no jacobians
      return l;
    } else {
      const auto n2 = n * n, th2 = th * th, tmp = n2 * n2 / th2;
      const T J_scale = max<T>(std::numeric_limits<T>::min(), T(1.0) / (tmp + T(1.0)));
      if constexpr (std::is_same_v<ExportJ, bool>)
        return std::make_pair(l, J_scale);
      else
        return std::make_pair(l, (J_scale * Jx_or_bool).eval());
    }
  }
}

/// @brief Arctan: Return a scaled norm of 'x' such that
/// @f$ n' = \sqrt(th * \atan(||x||² / th)) @f$,
/// also return the scale or scaled jacobian if given as input `Jx_or_bool`
template <typename T, typename ExportJ = std::nullptr_t>
auto ArctanNorm(const T &x, typename traits::params_trait<T>::Scalar th,
                const ExportJ &Jx_or_bool = nullptr) {
  return Arctan(L2(x, Jx_or_bool), th);
}

///////////////////// Cauchy ////////////////////////

/// @brief Cauchy: Return a scaled norm
/// @f$ n' = th * \sqrt(\log(1 + n² / th²)) @f$,
/// also return the scale or scaled jacobian if given as input `Jx_or_bool`
template <typename T, typename ExportJ = std::nullptr_t>
auto Cauchy(const T &n, typename traits::params_trait<T>::Scalar th,
            const ExportJ &Jx_or_bool = nullptr) {
  if constexpr (traits::is_pair_v<T>) {  // pair (norm, jacobian)
    return Cauchy(n.first, th, n.second);
  } else {
    using std::log;
    using std::sqrt;
    static_assert(!traits::is_matrix_or_array_v<T>, "`n` must be a scalar");
    const auto n2 = n * n, th2 = th * th;
    const auto s = T(1.0) + n2 / th2;
    const auto l = th * sqrt(log(s));
    if constexpr (std::is_null_pointer_v<ExportJ>) {  // no jacobians
      return l;
    } else {
      const T J_scale = std::max<T>(std::numeric_limits<T>::min(), T(1.0) / s);
      if constexpr (std::is_same_v<ExportJ, bool>)
        return std::make_pair(l, J_scale);
      else
        return std::make_pair(l, (J_scale * Jx_or_bool).eval());
    }
  }
}

/// @brief Cauchy: Return a scaled norm of 'x' such that
/// @f$ n' = th * \sqrt(\log(1 + ||x||² / th²)) @f$,
/// also return the scale or scaled jacobian if given as input `Jx_or_bool`
template <typename T, typename ExportJ = std::nullptr_t>
auto CauchyNorm(const T &x, typename traits::params_trait<T>::Scalar th,
                const ExportJ &Jx_or_bool = nullptr) {
  return Cauchy(L2(x, Jx_or_bool), th);
}

///////////////////// GemanMcClure ////////////////////////

/// @brief GemanMcClure: Return a scaled norm
/// @f$ n' = n / \sqrt(||x||² + th²) @f$,
/// also return the scale or scaled jacobian if given as input `Jx_or_bool`
template <typename T, typename ExportJ = std::nullptr_t>
auto GemanMcClure(const T &n, typename traits::params_trait<T>::Scalar th,
                  const ExportJ &Jx_or_bool = nullptr) {
  if constexpr (traits::is_pair_v<T>) {  // pair (norm, jacobian)
    return GemanMcClure(n.first, th, n.second);
  } else {
    using std::log;
    using std::sqrt;
    static_assert(!traits::is_matrix_or_array_v<T>, "`n` must be a scalar");
    const auto n2 = n * n, th2 = th * th;
    const auto e2_th2 = n2 + th2;
    const auto l = n / sqrt(e2_th2);
    if constexpr (std::is_null_pointer_v<ExportJ>) {  // no jacobians
      return l;
    } else {
      const auto J_scale = th2 / (e2_th2 * e2_th2);
      if constexpr (std::is_same_v<ExportJ, bool>)
        return std::make_pair(l, J_scale);
      else
        return std::make_pair(l, (J_scale * Jx_or_bool).eval());
    }
  }
}

/// @brief GemanMcClure: Return a scaled norm of 'x' such that
/// @f$ n' = ||x|| / \sqrt(||x||² + th²) @f$,
/// also return the scale or scaled jacobian if given as input `Jx_or_bool`
template <typename T, typename ExportJ = std::nullptr_t>
auto GemanMcClureNorm(const T &x, typename traits::params_trait<T>::Scalar th,
                      const ExportJ &Jx_or_bool = nullptr) {
  return GemanMcClure(L2(x, Jx_or_bool), th);
}

///////////////////// BlakeZisserman ////////////////////////

/// @brief BlakeZisserman: Return a scaled norm `n'` such that
/// @f$ n'² = -\log(\exp(-n²) + \exp(-th²))  @f$,
/// also return the scale or scaled jacobian if given as input `Jx_or_bool`
template <typename T, typename ExportJ = std::nullptr_t>
auto BlakeZisserman(const T &n, typename traits::params_trait<T>::Scalar th,
                    const ExportJ &Jx_or_bool = nullptr) {
  if constexpr (traits::is_pair_v<T>) {  // pair (norm, jacobian)
    return BlakeZisserman(n.first, th, n.second);
  } else {
    using std::exp;
    using std::log;
    using std::sqrt;
    static_assert(!traits::is_matrix_or_array_v<T>, "`n` must be a scalar");
    const auto n2 = n * n, th2 = th * th;
    const auto epsilon = exp(-th2);  // todo move this out
    const auto l = sqrt(-log(exp(-n2) + epsilon));
    if constexpr (std::is_null_pointer_v<ExportJ>) {  // no jacobians
      return l;
    } else {
      const auto J_scale = T(1.0) / (epsilon * exp(n2) + T(1.0));
      if constexpr (std::is_same_v<ExportJ, bool>)
        return std::make_pair(l, J_scale);
      else
        return std::make_pair(l, (J_scale * Jx_or_bool).eval());
    }
  }
}

/// @brief BlakeZisserman: Return a scaled norm of 'x' such that
/// @f$ n'² = -\log(\exp(-||x||²) + \exp(-th²))  @f$,
/// also return the scale or scaled jacobian if given as input `Jx_or_bool`
template <typename T, typename ExportJ = std::nullptr_t>
auto BlakeZissermanNorm(const T &x, typename traits::params_trait<T>::Scalar th,
                        const ExportJ &Jx_or_bool = nullptr) {
  return BlakeZisserman(L2(x, Jx_or_bool), th);
}

/** @} */

}  // namespace tinyopt::losses
