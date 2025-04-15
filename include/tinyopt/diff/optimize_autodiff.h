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

#include <cassert>
#include <cstddef>
#include <type_traits>
#include <utility>

#include <tinyopt/math.h>
#include <tinyopt/traits.h>  // must be before jet.h

#include <tinyopt/diff/jet.h>  // Import Jet's Automatic Differentiation

namespace tinyopt {

template <bool IsNLLS, typename X_t, typename ResidualsFunc, typename OptimizeFunc,
          typename OptionsType>
inline auto OptimizeWithAutoDiff(X_t &X, const ResidualsFunc &residuals,
                                 const OptimizeFunc &optimize, const OptionsType &options) {
  using ptrait = traits::params_trait<X_t>;
  using Scalar = typename ptrait::Scalar;
  constexpr Index Dims = ptrait::Dims;
  constexpr bool is_userdef_type =
      !std::is_floating_point_v<X_t> && !traits::is_matrix_or_array_v<X_t>;

  Index size = Dims;
  if constexpr (Dims == Dynamic) size = ptrait::dims(X);

  // Construct the Jet
  using Jet = diff::Jet<Scalar, Dims>;
  // XJetType is either of {Jet, Vector<Jet, N> or X_t::cast<Jet>()}
  using XJetType = std::conditional_t<std::is_floating_point_v<X_t>, Jet,
                                      decltype(ptrait::template cast<Jet>(X))>;
  // DXJetType is either of {nullptr, Vector<Jet, Dims>, Matrix<Jet, Rows, Cols>}
  using DXJetType = std::conditional_t<is_userdef_type, Vector<Jet, Dims>, std::nullptr_t>;
  XJetType x_jet;
  DXJetType dx_jet;  // only for user defined X type

  // Copy X to Jet values
  if constexpr (is_userdef_type) {  // X is user defined object
    dx_jet = DXJetType::Zero(size);
    for (Index i = 0; i < size; ++i) {
      // If X size at compile time is not known, we need to set the Jet.v
      if constexpr (Dims == Dynamic) dx_jet[i].v = Vector<Scalar, Dynamic>::Zero(size);
      dx_jet[i].v[i] = 1;
    }
    // dx_jet is constant
  } else if constexpr (std::is_floating_point_v<X_t>) {  // X is scalar
    x_jet = XJetType(X);
    x_jet.v[0] = 1;
  } else {  // X is a Vector or Matrix
    x_jet = ptrait::template cast<Jet>(X);
    // Set Jet's v
    for (int c = 0; c < X.cols(); ++c) {
      for (int r = 0; r < X.rows(); ++r) {
        const auto i = r + c * X.rows();
        if constexpr (Dims == Dynamic) x_jet(r, c).v = Vector<Scalar, Dims>::Zero(size);
        x_jet(r, c).v[i] = 1;
      }
    }
  }

  auto acc = [&](const auto &x, auto &grad, auto &H) {
    constexpr bool HasGrad = !traits::is_nullptr_v<decltype(grad)>;
    constexpr bool HasH = !traits::is_nullptr_v<decltype(H)>;
    // Update jet with latest 'x' values
    if constexpr (is_userdef_type) {          // X is user defined object
      x_jet = ptrait::template cast<Jet>(X);  // Cast X to a Jet type
      using ptrait_jet = traits::params_trait<XJetType>;
      ptrait_jet::PlusEq(x_jet, dx_jet);
    } else if constexpr (std::is_floating_point_v<X_t>) {  // X is scalar
      x_jet.a = x;
    } else {  // X is a Vector or Matrix
      for (int c = 0; c < x.cols(); ++c) {
        for (int r = 0; r < x.rows(); ++r) {
          x_jet(r, c).a = x(r, c);
        }
      }
    }

    // Retrieve the residuals
    const auto res = residuals(x_jet);
    using ResType = typename std::decay_t<decltype(res)>;

    static_assert(IsNLLS || traits::is_scalar_v<ResType>,
                  "General optimization cost function must return a scalar value");

    // Make sure the return type is either a Jet or Matrix/Array<Jet>
    static_assert(
        traits::is_jet_type_v<ResType> ||
        (traits::is_matrix_or_array_v<ResType> && traits::is_jet_type_v<typename ResType::Scalar>));

    if constexpr (traits::is_scalar_v<ResType>) {  // One residual
      if constexpr (HasGrad) {
        // Update H and gradient
        const auto &J = res.v;
        if constexpr (std::is_floating_point_v<X_t>) {
          grad[0] = J[0] * res.a;
          if constexpr (HasH) H(0, 0) = J[0] * J[0];
        } else {
          grad = J.transpose() * res.a;
          if constexpr (HasH) H = J * J.transpose();
        }
      }
      return IsNLLS ? res.a * res.a : res.a;  // NLLS -> return ε², else ε
    } else {                                  // Extract jacobian (TODO speed this up)
      constexpr int ResDims = traits::params_trait<ResType>::Dims;
      Index res_size = ResDims;
      if constexpr (ResDims == Dynamic) res_size = res.size();
      using J_t = Matrix<Scalar, ResDims, Dims>;

      J_t J(res_size, size);
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
        // Update H and gradient
        grad = J.transpose() * res_f;
        if constexpr (HasH) H = J.transpose() * J;
        // Logging of J
        if (options.log.enable && options.log.print_J_jet)
          TINYOPT_LOG("Jt:\n{}\n", J.transpose().eval());
      }
      // Returns the squared norm + number of residuals
      return std::make_pair(res_f.squaredNorm(), res_size);
    }
  };

  return optimize(X, acc, options);
}
}  // namespace tinyopt
