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

#include <tinyopt/cost.h>
#include <tinyopt/math.h>

#include <tinyopt/diff/jet.h>

namespace tinyopt::diff {

/// Return the function `f` residuals with the jacobian d f(x)/d(x) around `x` using
/// automatic differentiation
template <typename X_t, typename CostOrResFunc>
auto Eval(const X_t &x, const CostOrResFunc &cost_or_res_func) {
  using ptrait = traits::params_trait<X_t>;
  using Scalar = typename ptrait::Scalar;
  constexpr Index Dims = ptrait::Dims;
  constexpr bool is_userdef_type =
      !std::is_floating_point_v<X_t> && !traits::is_matrix_or_array_v<X_t>;
  const Index dims = traits::DynDims(x);

  // Construct the Jet
  using Jet = diff::Jet<Scalar, Dims>;
  // XJetType is either of {Jet, Vector<Jet, N> or X_t::cast<Jet>()}
  using XJetType = std::conditional_t<std::is_floating_point_v<X_t>, Jet,
                                      decltype(ptrait::template cast<Jet>(x))>;
  // DXJetType is either of {nullptr, Vector<Jet, Size>, Matrix<Jet, Rows,
  // Cols>}
  using DXJetType = std::conditional_t<is_userdef_type, Vector<Jet, Dims>, std::nullptr_t>;
  XJetType x_jet;
  DXJetType dx_jet;  // only for user defined X type

  // Copy X to Jet values
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
  // Update jet with latest 'x' values
  if constexpr (is_userdef_type) {          // X is user defined object
    x_jet = ptrait::template cast<Jet>(x);  // Cast X to a Jet type
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
  const auto res = cost_or_res_func(x_jet);
  using ResType = typename std::decay_t<decltype(res)>;

  // Make sure the return type is either a Jet or Matrix/Array<Jet>
  static_assert(
      traits::is_jet_type_v<ResType> ||
      (traits::is_matrix_or_array_v<ResType> && traits::is_jet_type_v<typename ResType::Scalar>));

  if constexpr (!traits::is_matrix_or_array_v<ResType>) {  // One residual
    return std::make_pair(res.a, res.v.transpose().eval());
  } else {
    constexpr int ResDims = traits::params_trait<ResType>::Dims;
    const Index res_dims = traits::DynDims(res);

    Matrix<Scalar, ResDims, Dims> J(res_dims, dims);
    Vector<Scalar, ResDims> res_f(res.size());
    if constexpr (traits::is_matrix_or_array_v<ResType>) {
      if constexpr (ResType::ColsAtCompileTime != 1) {  // Matrix or Vector with dynamic size
        for (int c = 0; c < res.cols(); ++c)
          for (int r = 0; r < res.rows(); ++r) {
            const Index i = r + c * res.rows();
            res_f[i] = res(r, c).a;
            J.row(i) = res(r, c).v;
          }
      } else {  // Vector
        for (Index i = 0; i < res_dims; ++i) {
          res_f[i] = res[i].a;
          J.row(i) = res[i].v;
        }
      }
    } else {  // scalar
      for (Index i = 0; i < res_dims; ++i) {
        res_f[i] = res.a;
        J.row(i) = res.v;
      }
    }
    return std::make_pair(res_f, J);
  }
}

/// Estimate the jacobian of d f(x)/d(x) around `x` using automatic
/// differentiation
template <typename X_t, typename CostOrResFunc>
auto CalculateJac(const X_t &x, const CostOrResFunc &cost_func) {
  const auto &[res, J] = Eval(x, cost_func);
  return J;
}

}  // namespace tinyopt::diff
