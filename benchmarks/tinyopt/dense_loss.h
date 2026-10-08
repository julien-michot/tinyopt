// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tinyopt/tinyopt.h>

namespace tinyopt::benchmark {

namespace dense_math_detail {

template <int Dimensions>
struct LossEvaluator {
  template <typename Parameters, typename Gradient, typename Hessian>
  Cost operator()(const Parameters& x, Gradient& gradient, Hessian& hessian) const {
    using Scalar = typename Parameters::Scalar;
    using FixedParameters = Eigen::Matrix<Scalar, Dimensions, 1>;
    using FixedJacobian = Eigen::Matrix<Scalar, Dimensions, Dimensions>;
    const FixedParameters parameters = x;
    const FixedParameters residuals = DenseMathResiduals(parameters);

    if constexpr (!traits::is_nullptr_v<Gradient>) {
      FixedJacobian jacobian;
      DenseMathJacobian(parameters, jacobian);
      gradient.noalias() = jacobian.transpose() * residuals;
      if constexpr (!traits::is_nullptr_v<Hessian>) {
        hessian.template topLeftCorner<Dimensions, Dimensions>().noalias() =
            jacobian.transpose() * jacobian;
      }
    }
    return Cost(Scalar(0.5) * residuals.squaredNorm(), Dimensions);
  }
};

}  // namespace dense_math_detail

struct DenseMathLoss {
  template <typename Parameters, typename Gradient, typename Hessian>
  Cost operator()(const Parameters& x, Gradient& gradient, Hessian& hessian) const {
    if constexpr (Parameters::RowsAtCompileTime == 1) {
      return dense_math_detail::LossEvaluator<1>{}(x, gradient, hessian);
    } else if constexpr (Parameters::RowsAtCompileTime == 2) {
      return dense_math_detail::LossEvaluator<2>{}(x, gradient, hessian);
    } else if constexpr (Parameters::RowsAtCompileTime == 3) {
      return dense_math_detail::LossEvaluator<3>{}(x, gradient, hessian);
    } else if (x.size() == 1) {
      return dense_math_detail::LossEvaluator<1>{}(x, gradient, hessian);
    } else if (x.size() == 2) {
      return dense_math_detail::LossEvaluator<2>{}(x, gradient, hessian);
    } else {
      return dense_math_detail::LossEvaluator<3>{}(x, gradient, hessian);
    }
  }
};

}  // namespace tinyopt::benchmark
