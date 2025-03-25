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

#include <tinyopt/jet.h>    // Import Ceres'Jet
#include <tinyopt/traits.h>

namespace tinyopt {

template <typename ParametersType, typename ResidualsFunc, typename OptimizeFunc, typename OptionsType>
inline auto OptimizeJet(ParametersType &X, ResidualsFunc &residuals,
                        const OptimizeFunc &optimize,
                        const OptionsType &options) {

  using ptrait = traits::params_trait<ParametersType>;
  using Scalar = ptrait::Scalar;
  constexpr int Size = ptrait::Dims;
  constexpr bool is_userdef_type = !std::is_floating_point_v<ParametersType>  && !traits::is_eigen_matrix_or_array_v<ParametersType>;

  const int size = ptrait::dims(X);
  // Construct the Jet
  using Jet = Jet<Scalar, Size>;
  // XJetType is either of {Jet, Vector<Jet, N> or ParametersType::cast<Jet>()}
  using XJetType = std::conditional_t<std::is_floating_point_v<ParametersType>, Jet,
                                      std::conditional_t<traits::is_eigen_matrix_or_array_v<ParametersType>,
                                        Vector<Jet, Size>, decltype(ptrait::template cast<Jet>(X))>>;
  // DXJetType is either of {nullptr, Vector<Jet, N>}
  using DXJetType = std::conditional_t<is_userdef_type, Vector<Jet, Size>, std::nullptr_t>;
  XJetType x_jet;
  DXJetType dx_jet; // only for user defined X type

  // Copy X to Jet values
  if constexpr (is_userdef_type) { // X is user defined object
    dx_jet = DXJetType::Zero(size);
    for (int i = 0; i < size; ++i) {
      dx_jet[i].v[i] = 1;
    }
    // dx_jet is constant
  } else if constexpr (std::is_floating_point_v<ParametersType>) { // X is scalar
    x_jet = XJetType(size);
    x_jet.v[0] = 1;
  } else { // X is a Vector
    x_jet = XJetType(size);
    for (int i = 0; i < size; ++i) {
      x_jet[i].v[i] = 1;
    }
  }

  auto acc = [&](const auto &x, auto &JtJ, auto &Jt_res) {

    // Update jet with latest 'x' values
    if constexpr (is_userdef_type) { // X is user defined object
      x_jet = ptrait::template cast<Jet>(X); // Cast X to a Jet type
      using ptrait_jet = traits::params_trait<XJetType>;
      ptrait_jet::pluseq(x_jet, dx_jet);
    } else if constexpr (std::is_floating_point_v<ParametersType>) { // X is scalar
      x_jet.a = x;
    } else { // X is a Vector
      for (int i = 0; i < size; ++i) {
        x_jet[i].a = x[i];
      }
    }

    // Retrieve the residuals
    const auto res = residuals(x_jet);
    using ResType =
        typename std::remove_const_t<std::remove_reference_t<decltype(res)>>;

    if constexpr (!traits::is_eigen_matrix_or_array_v<ResType> &&
                  std::is_floating_point_v<ParametersType>) {
      // Update JtJ and Jt*err
      const auto &J = res.v;
      JtJ(0, 0) = J[0] * J[0];
      Jt_res[0] = J[0] * res.a; // gradient
      // Return both the squared error and the number of residuals
      return std::make_pair(res.a * res.a, 1);
    } else { // Extract jacobian (TODO speed this up)
      constexpr int ResSize = traits::params_trait<ResType>::Dims;
      int res_size = ResSize; // System size (dynamic)
      if constexpr (ResSize != 1 && !std::is_floating_point_v<
                                        std::remove_reference_t<decltype(res)>>)
        res_size = res.size();

      // TODO avoid this copy
      Matrix<Scalar, ResSize, Size> J(res_size, size);
      for (int i = 0; i < res_size; ++i) {
        if constexpr (traits::is_eigen_matrix_or_array_v<ResType>)
          J.row(i) = res[i].v;
        else
          J.row(i) = res.v;
      }
      Vector<Scalar, ResSize> res_f(res.rows());
      for (int i = 0; i < res.rows(); ++i) {
        if constexpr (traits::is_eigen_matrix_or_array_v<ResType>)
          res_f[i] = res[i].a;
        else
          res_f[i] = res.a;
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

} // namespace tinyopt
