// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include <tinyopt/cost.h>
#include <tinyopt/math.h>
#include <tinyopt/traits.h>  // must be before jet.h

#include <tinyopt/diff/jet.h>  // Import Jet's Automatic Differentiation

namespace tinyopt {

/// @brief Optimize with automatic differentiation
/// @note This function is a slight? speedup compare to calling Optimize(tinyopt::diff::Eval)
template <bool IsNLLS, typename X_t, typename ResidualsFunc, typename OptimizeFunc,
          typename OptionsType>
inline auto OptimizeWithAutoDiff(X_t &x, const ResidualsFunc &residuals,
                                 const OptimizeFunc &optimize, const OptionsType &options) {
  using ptrait = traits::params_trait<X_t>;
  using Scalar = typename ptrait::Scalar;
  constexpr Index Dims = ptrait::Dims;
  constexpr bool is_userdef_type =
      !std::is_floating_point_v<X_t> && !traits::is_matrix_or_array_v<X_t>;

  const Index dims = traits::DynDims(x);

  // Construct the Jet
  using Jet = diff::Jet<Scalar, Dims>;

  // Note: XJetType and DXJetType are only instantiated to avoid having to set the 'v's at each
  // iteration

  // Construct the Jet for scalar or Matrix, so {Jet, Vector<Jet, N> ot nullptr_t}
  using XJetType = std::conditional_t<
      std::is_floating_point_v<X_t>, Jet,
      std::conditional_t<is_userdef_type, std::nullptr_t, decltype(ptrait::template cast<Jet>(x))>>;
  XJetType x_jet;

  // DXJetType is either of {Vector<Jet, Dims>, Matrix<Jet, Rows, Cols>, nullptr}
  using DXJetType = std::conditional_t<is_userdef_type, Vector<Jet, Dims>, std::nullptr_t>;

  DXJetType dx_jet;                 // only for user defined X type
  if constexpr (is_userdef_type) {  // X is user defined object
    dx_jet = DXJetType::Zero(dims);
    for (Index i = 0; i < dims; ++i) {
      // If X size at compile time is not known, we need to set the Jet.v
      if constexpr (Dims == Dynamic) dx_jet[i].v = Vector<Scalar, Dynamic>::Zero(dims);
      dx_jet[i].v[i] = 1;
    }
    // dx_jet is constant
  } else if constexpr (std::is_floating_point_v<X_t>) {  // X is scalar
    x_jet = XJetType(x);
    x_jet.v[0] = 1;
  } else {  // X is a Vector or Matrix
    x_jet = ptrait::template cast<Jet>(x);
    // Set Jet's v
    for (int c = 0; c < x.cols(); ++c) {
      for (int r = 0; r < x.rows(); ++r) {
        const auto i = r + c * x.rows();
        if constexpr (Dims == Dynamic) x_jet(r, c).v = Vector<Scalar, Dims>::Zero(dims);
        x_jet(r, c).v[i] = 1;
      }
    }
  }

  // Update jet with latest 'x' values, either returning the 'x_jet' or a local copy for user
  // defined types
  auto update_x_jet = [&](const auto &x) {
    if constexpr (is_userdef_type) {
      auto x_jet = ptrait::template cast<Jet>(x);  // Cast X to a Jet type // TODO reduce copy
      using ptrait_jet = traits::params_trait<std::decay_t<decltype(x_jet)>>;
      ptrait_jet::PlusEq(x_jet, dx_jet);
      return std::move(x_jet);
    } else if constexpr (std::is_floating_point_v<X_t>) {  // X is scalar
      x_jet.a = x;
      return x_jet;
    } else {  // X is a Vector or Matrix
      for (int c = 0; c < x.cols(); ++c) {
        for (int r = 0; r < x.rows(); ++r) {
          x_jet(r, c).a = x(r, c);
        }
      }
      return x_jet;
    }
  };

  auto acc = [&](const auto &x, auto &grad, auto &H) {
    using H_t = decltype(H);
    constexpr bool HasGrad = !traits::is_nullptr_v<decltype(grad)>;
    constexpr bool HasH = !traits::is_nullptr_v<H_t>;

    const auto &x_jet = update_x_jet(x);
    const auto res = residuals(x_jet);
    using ResType = typename std::decay_t<decltype(res)>;

    static_assert(IsNLLS || traits::is_scalar_v<ResType> || traits::is_tuple_v<ResType>,
                  "General optimization cost function must return a scalar value");

    auto process_tuple_item = [&](const auto &item) -> Cost {
      using ItemType = std::decay_t<decltype(item)>;

      if constexpr (traits::is_robust_residual_v<ItemType>) {
        const auto &raw = item.residual;
        using RawType = std::decay_t<decltype(raw)>;
        static_assert(traits::is_jet_type_v<RawType> ||
                      (traits::is_matrix_or_array_v<RawType> &&
                       traits::is_jet_type_v<typename RawType::Scalar>));

        constexpr int ResDims = traits::params_trait<RawType>::Dims;
        const Index res_size = traits::DynDims(raw);
        using J_t = Matrix<Scalar, ResDims, Dims>;

        J_t J(res_size, dims);
        Vector<Scalar, ResDims> res_f(res_size);
        if constexpr (traits::is_matrix_or_array_v<RawType>) {
          if constexpr (RawType::ColsAtCompileTime != 1) {
            for (int c = 0; c < raw.cols(); ++c)
              for (int r = 0; r < raw.rows(); ++r) {
                const Index i = r + c * raw.rows();
                J.row(i) = raw(r, c).v;
                res_f[i] = raw(r, c).a;
              }
          } else {
            for (Index i = 0; i < res_size; ++i) {
              J.row(i) = raw[i].v;
              res_f[i] = raw[i].a;
            }
          }
        } else {
          J.row(0) = raw.v;
          res_f[0] = raw.a;
        }

        const double sq_norm = static_cast<double>(res_f.squaredNorm());
        const double weight =
            sq_norm <= item.delta2
                ? 1.0
                : item.delta / std::sqrt(std::max(sq_norm, std::numeric_limits<double>::min()));
        const double robust_cost =
            sq_norm <= item.delta2
                ? sq_norm
                : 2.0 * item.delta *
                          std::sqrt(std::max(sq_norm, std::numeric_limits<double>::min())) -
                      item.delta2;

        if constexpr (HasGrad) grad.noalias() += weight * (J.transpose() * res_f);
        if constexpr (HasH) {
          if constexpr (traits::is_sparse_matrix_v<H_t>) {
            H += (weight * (J.transpose() * J)).sparseView();
          } else {
            H.noalias() += weight * (J.transpose() * J);
          }
        }

        return Cost(robust_cost, res_size, sq_norm <= item.delta2 ? 1.0f : 0.0f);
      } else {
        static_assert(
            traits::is_jet_type_v<ItemType> || (traits::is_matrix_or_array_v<ItemType> &&
                                                traits::is_jet_type_v<typename ItemType::Scalar>),
            "Residual blocks must be Jet-valued vectors/matrices or robust wrappers");

        if constexpr (traits::is_scalar_v<ItemType>) {
          if constexpr (HasGrad) {
            const auto &J = item.v;
            if constexpr (std::is_floating_point_v<X_t>) {
              grad[0] += J[0] * item.a;
              if constexpr (HasH) H(0, 0) += J[0] * J[0];
            } else {
              grad.noalias() += J.transpose() * item.a;
              if constexpr (HasH) {
                if constexpr (traits::is_sparse_matrix_v<H_t>) {
                  H += (J * J.transpose()).sparseView();
                } else {
                  H.noalias() += J * J.transpose();
                }
              }
            }
          }
          return Cost(IsNLLS ? item.a * item.a : item.a, 1);
        } else {
          constexpr int ResDims = traits::params_trait<ItemType>::Dims;
          const Index res_size = traits::DynDims(item);
          using J_t = Matrix<Scalar, ResDims, Dims>;

          J_t J(res_size, dims);
          Vector<Scalar, ResDims> res_f(res_size);
          if constexpr (traits::is_matrix_or_array_v<ItemType>) {
            if constexpr (ItemType::ColsAtCompileTime != 1) {
              for (int c = 0; c < item.cols(); ++c)
                for (int r = 0; r < item.rows(); ++r) {
                  const Index i = r + c * item.rows();
                  J.row(i) = item(r, c).v;
                  res_f[i] = item(r, c).a;
                }
            } else {
              for (Index i = 0; i < res_size; ++i) {
                J.row(i) = item[i].v;
                res_f[i] = item[i].a;
              }
            }
          }

          if constexpr (HasGrad) {
            grad.noalias() += J.transpose() * res_f;
            if constexpr (HasH) {
              if constexpr (traits::is_sparse_matrix_v<H_t>) {
                H += (J.transpose() * J).sparseView();
              } else {
                H.noalias() += J.transpose() * J;
              }
            }
          }
          return Cost(res_f.squaredNorm(), res_size);
        }
      }
    };

    if constexpr (traits::is_tuple_v<ResType>) {
      Cost total;
      std::apply([&](auto &&...items) { ((total += process_tuple_item(items)), ...); }, res);
      return total;
    } else {
      static_assert(IsNLLS || traits::is_scalar_v<ResType>,
                    "General optimization cost function must return a scalar value");
      static_assert(traits::is_jet_type_v<ResType> ||
                    (traits::is_matrix_or_array_v<ResType> &&
                     traits::is_jet_type_v<typename ResType::Scalar>));

      if constexpr (traits::is_scalar_v<ResType>) {  // One residual
        if constexpr (HasGrad) {
          const auto &J = res.v;
          if constexpr (std::is_floating_point_v<X_t>) {
            if constexpr (IsNLLS)
              grad[0] = J[0] * res.a;
            else
              grad[0] = J[0];
            if constexpr (HasH) H(0, 0) = J[0] * J[0];
          } else {
            if constexpr (IsNLLS)
              grad.noalias() = J.transpose() * res.a;
            else
              grad = J;
            if constexpr (HasH) {
              if constexpr (traits::is_sparse_matrix_v<H_t>) {
                H += (J * J.transpose()).sparseView();
              } else {
                H.noalias() += J * J.transpose();
              }
            }
          }
        }
        return Cost{IsNLLS ? res.a * res.a : res.a};  // NLLS -> return ε², else ε
      } else {                                        // Extract jacobian (TODO speed this up)
        constexpr int ResDims = traits::params_trait<ResType>::Dims;
        const Index res_size = traits::DynDims(res);
        using J_t = Matrix<Scalar, ResDims, Dims>;

        J_t J(res_size, dims);  // TODO make J sparse if H is.
        Vector<Scalar, ResDims> res_f(res.size());
        if constexpr (traits::is_matrix_or_array_v<ResType>) {
          if constexpr (ResType::ColsAtCompileTime != 1) {  // Matrix or Vector with dynamic size
            for (int c = 0; c < res.cols(); ++c)
              for (int r = 0; r < res.rows(); ++r) {
                const Index i = r + c * res.rows();
                if constexpr (HasGrad) J.row(i) = res(r, c).v;
                res_f[i] = res(r, c).a;
              }
          } else {  // Vector
            for (Index i = 0; i < res_size; ++i) {
              if constexpr (HasGrad) J.row(i) = res[i].v;
              res_f[i] = res[i].a;
            }
          }
        } else {  // scalar
          for (Index i = 0; i < res_size; ++i) {
            if constexpr (HasGrad) J.row(i) = res.v;
            res_f[i] = res.a;
          }
        }
        if constexpr (HasGrad) {
          grad = J.transpose() * res_f;
          if constexpr (HasH) {
            if constexpr (traits::is_sparse_matrix_v<H_t>) {
              H = (J.transpose() * J).sparseView();
            } else {
              H = J.transpose() * J;
            }
          }
          if (options.log.enable && options.log.print_J_jet)
            TINYOPT_LOG("Jt:\n{}\n", J.transpose().eval());
        }
        return Cost(res_f.squaredNorm(), res_size);
      }
    }
  };

  return optimize(x, acc, options);
}
}  // namespace tinyopt
