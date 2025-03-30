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
#include <utility>

#include <Eigen/src/Core/Matrix.h>
#include <Eigen/src/Core/util/Constants.h>
#include <tinyopt/traits.h> // must be before jet.h

#include <tinyopt/jet.h>  // Import Ceres'Jet

namespace tinyopt {

template <typename ParametersType, typename ResidualsFunc, typename OptimizeFunc,
          typename OptionsType>
inline auto OptimizeJet(ParametersType &X, const ResidualsFunc &residuals,
                        const OptimizeFunc &optimize, const OptionsType &options) {
  using ptrait = traits::params_trait<ParametersType>;
  using Scalar = typename ptrait::Scalar;
  constexpr int Size = ptrait::Dims;
  constexpr bool is_userdef_type =
      !std::is_floating_point_v<ParametersType> && !traits::is_matrix_or_array_v<ParametersType>;

  int size = Size;
  if constexpr (Size == Dynamic) size = ptrait::dims(X);

  // Construct the Jet
  using Jet = Jet<Scalar, Size>;
  // XJetType is either of {Jet, Vector<Jet, N> or ParametersType::cast<Jet>()}
  using XJetType = std::conditional_t<std::is_floating_point_v<ParametersType>, Jet,
                                      decltype(ptrait::template cast<Jet>(X))>;
  // DXJetType is either of {nullptr, Vector<Jet, Size>, Matrix<Jet, Rows, Cols>}
  using DXJetType = std::conditional_t<is_userdef_type, Vector<Jet, Size>, std::nullptr_t>;
  XJetType x_jet;
  DXJetType dx_jet;  // only for user defined X type

  // Copy X to Jet values
  if constexpr (is_userdef_type) {  // X is user defined object
    dx_jet = DXJetType::Zero(size);
    for (int i = 0; i < size; ++i) {
      // If X size at compile time is not known, we need to set the Jet.v
      if constexpr (Size == Dynamic) dx_jet[i].v = Vector<Scalar, Dynamic>::Zero(size);
      dx_jet[i].v[i] = 1;
    }
    // dx_jet is constant
  } else if constexpr (std::is_floating_point_v<ParametersType>) {  // X is scalar
    x_jet = XJetType(size);
    x_jet.v[0] = 1;
  } else {  // X is a Vector or Matrix
    x_jet = ptrait::template cast<Jet>(X);
    // Set Jet's v
    for (int c = 0; c < X.cols(); ++c) {
      for (int r = 0; r < X.rows(); ++r) {
        const int i = r + c * X.rows();
        if constexpr (Size == Dynamic) x_jet(r, c).v = Vector<Scalar, Size>::Zero(size);
        x_jet(r, c).v[i] = 1;
      }
    }
  }

  auto acc = [&](const auto &x, auto &JtJ, auto &Jt_res) {
    // Update jet with latest 'x' values
    if constexpr (is_userdef_type) {          // X is user defined object
      x_jet = ptrait::template cast<Jet>(X);  // Cast X to a Jet type
      using ptrait_jet = traits::params_trait<XJetType>;
      ptrait_jet::pluseq(x_jet, dx_jet);
    } else if constexpr (std::is_floating_point_v<ParametersType>) {  // X is scalar
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
    using ResType = typename std::remove_const_t<std::remove_reference_t<decltype(res)>>;

    // Make sure the return type is either a Jet or Matrix/Array<Jet>
    static_assert(
        traits::is_jet_type_v<ResType> ||
        (traits::is_matrix_or_array_v<ResType> && traits::is_jet_type_v<typename ResType::Scalar>));

    if constexpr (!traits::is_matrix_or_array_v<ResType>) {  // One residual
      // Update JtJ and Jt*err
      const auto &J = res.v;
      if constexpr (std::is_floating_point_v<ParametersType>) {
        JtJ(0, 0) = J[0] * J[0];
        Jt_res[0] = J[0] * res.a;
      } else {
        JtJ = J * J.transpose();
        Jt_res = J.transpose() * res.a;
      }
      // Return both the squared error and the number of residuals
      return std::make_pair(res.a * res.a, 1);
    } else {  // Extract jacobian (TODO speed this up)
      constexpr int ResSize = traits::params_trait<ResType>::Dims;
      int res_size = ResSize;  // System size (dynamic)
      if constexpr (ResSize != 1 &&
                    !std::is_floating_point_v<std::remove_reference_t<decltype(res)>>)
        res_size = res.size();

      // TODO avoid this copy
      Matrix<Scalar, ResSize, Size> J(res_size, size);
      Vector<Scalar, ResSize> res_f(res.size());
      if constexpr (traits::is_matrix_or_array_v<ResType>) {
        if constexpr (ResType::ColsAtCompileTime != 1) {  // Matrix or Vector with dynamic size
          for (int c = 0; c < res.cols(); ++c)
            for (int r = 0; r < res.rows(); ++r) {
              const int i = r + c * res.rows();
              J.row(i) = res(r, c).v;
              res_f[i] = res(r, c).a;
            }
        } else {  // Vector
          for (int i = 0; i < res_size; ++i) {
            J.row(i) = res[i].v;
            res_f[i] = res[i].a;
          }
        }
      } else {  // scalar
        for (int i = 0; i < res_size; ++i) {
          J.row(i) = res.v;
          res_f[i] = res.a;
        }
      }
      if (options.log.enable && options.log.print_J_jet) {
        TINYOPT_LOG("Jt:\n{}\n", J.transpose().eval());
      }
      // Update JtJ and Jt*err
      JtJ = J.transpose() * J;
      Jt_res = J.transpose() * res_f;
      // Returns the squared residuals norm
      return std::make_pair(res_f.squaredNorm(), res_size);
    }
  };

  return optimize(X, acc, options);
}

}  // namespace tinyopt
