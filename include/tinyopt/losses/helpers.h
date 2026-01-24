// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tinyopt/math.h>
#include <tinyopt/traits.h>

/***
 * @brief Macro helper to define a loss, see activations.h for examples.
 * Not sure it was a good idea but look at activations.h, it's sooo slim!
 */
#define DefineLoss(name, code, jac_code)                                                          \
  template <typename T, typename ExportJ>                                                         \
  auto name(const T &x, const ExportJ &Jx_or_bool);                                               \
  /* Loss that only returns the final loss */                                                     \
  template <typename T>                                                                           \
  auto name(const T &x) {                                                                         \
    if constexpr (traits::is_pair_v<T>) {                                                         \
      return name(x.first, x.second);                                                             \
    } else {                                                                                      \
      constexpr bool IsMatrix = traits::is_matrix_or_array_v<T> || traits::is_sparse_matrix_v<T>; \
      if constexpr (IsMatrix) { /* matrix */                                                      \
        using Scalar = typename T::Scalar;                                                        \
        return x.unaryExpr([](Scalar v) { return name<Scalar>(v); }).eval();                      \
      } else { /* scalar */                                                                       \
        using Scalar = std::decay_t<T>;                                                           \
        return Scalar((code));                                                                    \
      }                                                                                           \
    }                                                                                             \
  }                                                                                               \
  /* Loss  that returns both the loss and the Jacobian  */                                        \
  template <typename T, typename ExportJ>                                                         \
  auto name(const T &x, const ExportJ &Jx_or_bool) {                                              \
    constexpr bool HasJac = traits::is_matrix_or_scalar_v<std::decay_t<ExportJ>>;                 \
    constexpr bool IsMatrix = traits::is_matrix_or_array_v<T> || traits::is_sparse_matrix_v<T>;   \
    const auto l = name(x);                                                                       \
    if constexpr (IsMatrix) { /* matrix */                                                        \
      using Scalar = typename T::Scalar;                                                          \
      const auto J = ((jac_code));                                                                \
      if constexpr (HasJac) {                                                                     \
        return std::make_pair(l, (J * Jx_or_bool).matrix().eval());                               \
      } else {                                                                                    \
        constexpr int DimsJ = traits::params_trait<T>::Dims;                                      \
        return std::make_pair(l, Matrix<Scalar, DimsJ, DimsJ>(J));                                \
      }                                                                                           \
    } else { /* scalar */                                                                         \
      if constexpr (HasJac) {                                                                     \
        return std::make_pair(l, (0 /*TODO*/ * Jx_or_bool).matrix().eval());                      \
      } else {                                                                                    \
        return std::make_pair(l, 0 /*TODO*/);                                                     \
      }                                                                                           \
    }                                                                                             \
  }

/***
 * @brief Macro helper to define a loss with a parameter `a`, see activations.h for examples
 */
#define DefineLoss2(name, code, jac_code)                                                         \
  template <typename T, typename ExportJ, typename ParamType = float>                             \
  auto name(const T &x, ParamType a, const ExportJ &Jx_or_bool);                                  \
  /* Loss that only returns the final loss */                                                     \
  template <typename T, typename ParamType = float>                                               \
  auto name(const T &x, ParamType a) {                                                            \
    if constexpr (traits::is_pair_v<T>) {                                                         \
      return name(x.first, x.second, a);                                                          \
    } else {                                                                                      \
      constexpr bool IsMatrix = traits::is_matrix_or_array_v<T> || traits::is_sparse_matrix_v<T>; \
      if constexpr (IsMatrix) { /* matrix */                                                      \
        using Scalar = typename T::Scalar;                                                        \
        return x.unaryExpr([a](Scalar v) -> Scalar { return name<Scalar>(v, a); }).eval();        \
      } else { /* scalar */                                                                       \
        using Scalar = std::decay_t<T>;                                                           \
        return Scalar((code));                                                                    \
      }                                                                                           \
    }                                                                                             \
  }                                                                                               \
  /* Loss  that returns both the loss and the Jacobian  */                                        \
  template <typename T, typename ExportJ, typename ParamType>                                     \
  auto name(const T &x, ParamType a, const ExportJ &Jx_or_bool) {                                 \
    constexpr bool HasJac = traits::is_matrix_or_scalar_v<std::decay_t<ExportJ>>;                 \
    constexpr bool IsMatrix = traits::is_matrix_or_array_v<T> || traits::is_sparse_matrix_v<T>;   \
    const auto l = name(x, a);                                                                    \
    if constexpr (IsMatrix) { /* matrix */                                                        \
      using Scalar = typename T::Scalar;                                                          \
      const auto J = ((jac_code));                                                                \
      if constexpr (HasJac) {                                                                     \
        return std::make_pair(l, (J * Jx_or_bool).matrix().eval());                               \
      } else {                                                                                    \
        constexpr int DimsJ = traits::params_trait<T>::Dims;                                      \
        return std::make_pair(l, Matrix<Scalar, DimsJ, DimsJ>(J));                                \
      }                                                                                           \
    } else { /* scalar */                                                                         \
      if constexpr (HasJac) {                                                                     \
        return std::make_pair(l, ((0 /*TODO*/) * Jx_or_bool).matrix().eval());                    \
      } else {                                                                                    \
        return std::make_pair(l, 0 /*TODO*/);                                                     \
      }                                                                                           \
    }                                                                                             \
  }
