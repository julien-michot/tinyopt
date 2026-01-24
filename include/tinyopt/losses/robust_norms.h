// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

/// If you land here, put your shades on because those fat functions are still hurting my eyes as
/// of today.

#pragma once

#include <tinyopt/math.h>
#include <tinyopt/traits.h>

#include <tinyopt/losses/norms.h>

namespace tinyopt::losses {

/**
 * @name M-Estimators / Robust Losses
 *
 * @brief These functions return a loss and a scale to apply on the jacobian or the scaled jacobian
 *
 * The loss's gradient is then s*res.t(), with 's' the scale returned by the robust function
 * (e.g. Truncated, huber, Artcan, etc.), typically 1 if the ||res|| < threshold.
 * The scale can then be used to solve the lossal equations: JtJ * dx = Jt*res*s
 * Ex: Huber : s = {1, th/||x||} so that we thereby clip the gradient to a max of theshold 'th'
 * while keeping the direction. grdaient = {inlier:Jt * res; outlier:Jt * res * th / ||res||}
 *
 * @{
 */

///////////////////// TruncatedL2 ////////////////////////

/// @brief Hard Truncation: Clip the given loss `n` to a max of `th`,
/// also return the scale {0 1} or scaled jacobian if given as input `Jx_or_bool`
template <typename T, typename ExportJ = std::nullptr_t>
auto Truncated(const T &n2, typename traits::params_trait<T>::Scalar th2,
               const ExportJ &Jx_or_bool = nullptr) {
  if constexpr (traits::is_pair_v<T>) {  // pair (loss, jacobian)
    return Truncated(n2.first, th2, n2.second);
  } else {
    static_assert(!traits::is_matrix_or_array_v<T>, "`n` must be a scalar");
    const auto l = n2 <= th2 ? n2 : T(th2);
    if constexpr (std::is_null_pointer_v<ExportJ>) {
      return l;
    } else if constexpr (traits::is_bool_v<ExportJ>) {
      return std::make_pair(l, n2 <= th2 ? T(1) : T(0));
    } else {  // Jx_or_bool is Jx
      assert(Jx_or_bool.rows() == 1);
      if (n2 <= th2) {
        return std::make_pair(l, Jx_or_bool);
      } else {
        return std::make_pair(l, ExportJ::Zero(1, Jx_or_bool.cols()).eval());
      }
    }
  }
}

/// @brief Hard Truncated L2 Loss: Clip the L2 loss of `x` to a max of `th`,
/// also return the scale {0, 1} or scaled jacobian if given as input `Jx_or_bool`
template <typename T, typename ExportJ = std::nullptr_t>
auto TruncatedLoss(const T &x, typename traits::params_trait<T>::Scalar th2,
                   const ExportJ &Jx_or_bool = nullptr) {
  return Truncated(SquaredL2(x, Jx_or_bool), th2);
}

///////////////////// Huber ////////////////////////

/// @brief Huber: Return a scaled loss @f$ n'= n if n < th, else n'= \sqrt(2.0 * th * n - th²) @f$,
/// also return the scale {1, th/n} or scaled jacobian if given as input `Jx_or_bool`
template <typename T, typename ExportJ = std::nullptr_t>
auto Huber(const T &n2, typename traits::params_trait<T>::Scalar th2,
           const ExportJ &Jx_or_bool = nullptr) {
  if constexpr (traits::is_pair_v<T>) {  // pair (loss, jacobian)
    return Huber(n2.first, th2, n2.second);
  } else {
    using std::max;
    using std::sqrt;
    static_assert(!traits::is_matrix_or_array_v<T>, "`n` must be a scalar");

    if (n2 <= th2) {  // Inlier
      const auto l = n2;
      if constexpr (std::is_null_pointer_v<ExportJ>) {  // no jacobians
        return l;
      } else {
        if constexpr (traits::is_bool_v<ExportJ>)
          return std::make_pair(l, T(1));
        else
          return std::make_pair(l, Jx_or_bool);
      }
    } else {  // Outlier
      const auto th = T(sqrt(th2)), n = sqrt(n2);
      const auto l = T(2.0) * th * n - T(th2);
      if constexpr (std::is_null_pointer_v<ExportJ>) {  // no jacobians
        return l;
      } else {
        const T J_scale = max<T>(std::numeric_limits<T>::min(), th / n);
        if constexpr (traits::is_bool_v<ExportJ>)
          return std::make_pair(l, J_scale);
        else
          return std::make_pair(l, (J_scale * Jx_or_bool).eval());
      }
    }
  }
}

/// @brief Huber: Return a scaled loss of 'x' such that
/// @f$ n'² = ||x||² if ||x||² < th2, else n'² = 2.0 * th * n - th² @f$,
/// also return the scale {1, th/n} or scaled jacobian if given as input
/// `Jx_or_bool`
/// @note The gradient is then {inlier:x.t(), outlier:x.t() * s}, with @f$s = th / ||x||@f$.
template <typename T, typename ExportJ = std::nullptr_t>
auto HuberLoss(const T &x, typename traits::params_trait<T>::Scalar th2,
               const ExportJ &Jx_or_bool = nullptr) {
  return Huber(SquaredL2(x, Jx_or_bool), th2);
}

///////////////////// Tuckey ////////////////////////

/// @brief Tukey: return a scaled loss (without the /6)
/// @f$ n'² = n² if n < th, else n'² = 2.0 * h * n - th² @f$,
/// also return the scale or scaled jacobian if given as input `Jx_or_bool`
template <typename T, typename ExportJ = std::nullptr_t>
auto Tukey(const T &n2, typename traits::params_trait<T>::Scalar th2,
           const ExportJ &Jx_or_bool = nullptr) {
  if constexpr (traits::is_pair_v<T>) {  // pair (loss, jacobian)
    return Tukey(n2.first, th2, n2.second);
  } else {
    static_assert(!traits::is_matrix_or_array_v<T>, "`n` must be a scalar");
    if (n2 <= th2) {  // Inlier
      const auto s = T(1.0) - n2 / T(th2), s2 = s * s;
      const auto l = T(th2) * (T(1.0) - s2 * s);
      if constexpr (std::is_null_pointer_v<ExportJ>) {  // no jacobians
        return l;
      } else {
        const auto J_scale = T(3.0) * (th2 - n2) * (th2 - n2) / (th2 * th2);
        if constexpr (traits::is_bool_v<ExportJ>)
          return std::make_pair(l, J_scale);
        else
          return std::make_pair(l, (J_scale * Jx_or_bool).eval());
      }
    } else {  // Outlier
      const T l = T(th2);
      if constexpr (std::is_null_pointer_v<ExportJ>) {  // no jacobians
        return l;
      } else {
        if constexpr (traits::is_bool_v<ExportJ>)
          return std::make_pair(l, T(0));
        else
          return std::make_pair(l, ExportJ::Zero(1, Jx_or_bool.cols()).eval());
      }
    }
  }
}

/// @brief Tukey: Return a scaled loss of 'x' such that
/// @f$ n'² = ||x||² if ||x|| < th, else n'² = 2.0 * h * n - th² @f$,
/// also return the scale or scaled jacobian if given as input `Jx_or_bool`
template <typename T, typename ExportJ = std::nullptr_t>
auto TukeyLoss(const T &x, typename traits::params_trait<T>::Scalar th2,
               const ExportJ &Jx_or_bool = nullptr) {
  return Tukey(SquaredL2(x, Jx_or_bool), th2);
}

///////////////////// Arctan ////////////////////////

/// @brief Arctan: Return a scaled loss
/// @f$ n'² = th * \atan(n² / th) @f$,
/// also return the scale or scaled jacobian if given as input `Jx_or_bool`
template <typename T, typename ExportJ = std::nullptr_t>
auto Arctan(const T &n2, typename traits::params_trait<T>::Scalar th2,
            const ExportJ &Jx_or_bool = nullptr) {
  if constexpr (traits::is_pair_v<T>) {  // pair (loss, jacobian)
    return Arctan(n2.first, th2, n2.second);
  } else {
    using std::atan2;
    using std::max;
    using std::sqrt;
    static_assert(!traits::is_matrix_or_array_v<T>, "`n` must be a scalar");
    const T th = T(sqrt(th2));
    const auto l = th * atan2(n2, th);
    if constexpr (std::is_null_pointer_v<ExportJ>) {  // no jacobians
      return l;
    } else {
      const auto tmp = n2 * n2 / th2;
      const T J_scale = max<T>(std::numeric_limits<T>::min(), T(1.0) / (tmp + T(1.0)));
      if constexpr (traits::is_bool_v<ExportJ>)
        return std::make_pair(l, J_scale);
      else
        return std::make_pair(l, (J_scale * Jx_or_bool).eval());
    }
  }
}

/// @brief Arctan: Return a scaled loss of 'x' such that
/// @f$ n'² = th * \atan(||x||² / th) @f$,
/// also return the scale or scaled jacobian if given as input `Jx_or_bool`
template <typename T, typename ExportJ = std::nullptr_t>
auto ArctanLoss(const T &x, typename traits::params_trait<T>::Scalar th2,
                const ExportJ &Jx_or_bool = nullptr) {
  return Arctan(SquaredL2(x, Jx_or_bool), th2);
}

///////////////////// Cauchy ////////////////////////

/// @brief Cauchy: Return a scaled loss
/// @f$ n'² = th² * \log(1 + n² / th²) @f$,
/// also return the scale or scaled jacobian if given as input `Jx_or_bool`
template <typename T, typename ExportJ = std::nullptr_t>
auto Cauchy(const T &n2, typename traits::params_trait<T>::Scalar th2,
            const ExportJ &Jx_or_bool = nullptr) {
  if constexpr (traits::is_pair_v<T>) {  // pair (loss, jacobian)
    return Cauchy(n2.first, th2, n2.second);
  } else {
    using std::log;
    using std::sqrt;
    static_assert(!traits::is_matrix_or_array_v<T>, "`n` must be a scalar");
    const auto s = T(1.0) + n2 / T(th2);
    const auto l = T(th2) * log(s);
    if constexpr (std::is_null_pointer_v<ExportJ>) {  // no jacobians
      return l;
    } else {
      const T J_scale = std::max<T>(std::numeric_limits<T>::min(), T(1.0) / s);
      if constexpr (traits::is_bool_v<ExportJ>)
        return std::make_pair(l, J_scale);
      else
        return std::make_pair(l, (J_scale * Jx_or_bool).eval());
    }
  }
}

/// @brief Cauchy: Return a scaled loss of 'x' such that
/// @f$ n'² = th² * \log(1 + ||x||² / th²) @f$,
/// also return the scale or scaled jacobian if given as input `Jx_or_bool`
template <typename T, typename ExportJ = std::nullptr_t>
auto CauchyLoss(const T &x, typename traits::params_trait<T>::Scalar th2,
                const ExportJ &Jx_or_bool = nullptr) {
  return Cauchy(SquaredL2(x, Jx_or_bool), th2);
}

///////////////////// GemanMcClure ////////////////////////

/// @brief GemanMcClure: Return a scaled loss
/// @f$ n'² = n² / (||x||² + th²) @f$,
/// also return the scale or scaled jacobian if given as input `Jx_or_bool`
template <typename T, typename ExportJ = std::nullptr_t>
auto GemanMcClure(const T &n2, typename traits::params_trait<T>::Scalar th2,
                  const ExportJ &Jx_or_bool = nullptr) {
  if constexpr (traits::is_pair_v<T>) {  // pair (loss, jacobian)
    return GemanMcClure(n2.first, th2, n2.second);
  } else {
    using std::log;
    using std::sqrt;
    static_assert(!traits::is_matrix_or_array_v<T>, "`n` must be a scalar");
    const T e2_th2 = n2 + T(th2);
    const auto l = n2 / (e2_th2);
    if constexpr (std::is_null_pointer_v<ExportJ>) {  // no jacobians
      return l;
    } else {
      const auto J_scale = th2 / (e2_th2 * e2_th2);
      if constexpr (traits::is_bool_v<ExportJ>)
        return std::make_pair(l, J_scale);
      else
        return std::make_pair(l, (J_scale * Jx_or_bool).eval());
    }
  }
}

/// @brief GemanMcClure: Return a scaled loss of 'x' such that
/// @f$ n' = ||x|| / \sqrt(||x||² + th²) @f$,
/// also return the scale or scaled jacobian if given as input `Jx_or_bool`
template <typename T, typename ExportJ = std::nullptr_t>
auto GemanMcClureLoss(const T &x, typename traits::params_trait<T>::Scalar th2,
                      const ExportJ &Jx_or_bool = nullptr) {
  return GemanMcClure(SquaredL2(x, Jx_or_bool), th2);
}

///////////////////// BlakeZisserman ////////////////////////

/// @brief BlakeZisserman: Return a scaled loss `n'` such that
/// @f$ n'² = -\log(\exp(-n²) + \exp(-th²))  @f$,
/// also return the scale or scaled jacobian if given as input `Jx_or_bool`
template <typename T, typename ExportJ = std::nullptr_t>
auto BlakeZisserman(const T &n2, typename traits::params_trait<T>::Scalar th2,
                    const ExportJ &Jx_or_bool = nullptr) {
  if constexpr (traits::is_pair_v<T>) {  // pair (loss, jacobian)
    return BlakeZisserman(n2.first, th2, n2.second);
  } else {
    using std::exp;
    using std::log;
    using std::sqrt;
    static_assert(!traits::is_matrix_or_array_v<T>, "`n` must be a scalar");
    const auto epsilon = T(exp(-th2));
    const auto l = -log(exp(-n2) + epsilon);
    if constexpr (std::is_null_pointer_v<ExportJ>) {  // no jacobians
      return l;
    } else {
      const auto J_scale = T(1.0) / (epsilon * exp(n2) + T(1.0));
      if constexpr (traits::is_bool_v<ExportJ>)
        return std::make_pair(l, J_scale);
      else
        return std::make_pair(l, (J_scale * Jx_or_bool).eval());
    }
  }
}

/// @brief BlakeZisserman: Return a scaled loss of 'x' such that
/// @f$ n'² = -\log(\exp(-||x||²) + \exp(-th²))  @f$,
/// also return the scale or scaled jacobian if given as input `Jx_or_bool`
template <typename T, typename ExportJ = std::nullptr_t>
auto BlakeZissermanLoss(const T &x, typename traits::params_trait<T>::Scalar th2,
                        const ExportJ &Jx_or_bool = nullptr) {
  return BlakeZisserman(SquaredL2(x, Jx_or_bool), th2);
}

/** @} */

}  // namespace tinyopt::losses
