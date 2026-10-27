// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstddef>
#include <exception>
#include <stdexcept>
#include <utility>
#include <vector>

#include <tinyopt/c/c_api_common.h>
#include <tinyopt/optimize.h>
#include <tinyopt/optimizers/dl.h>
#include <tinyopt/optimizers/gn.h>
#include <tinyopt/optimizers/lm.h>

#include "c_api_options.h"
#include "c_api_params.h"
#include "c_api_problem.h"

namespace tinyopt::c_api_detail {

/// Adapts a C block-Hessian callback to Tinyopt's accumulation interface.
template <typename Scalar, typename Handle, typename Callback>
struct BlockHessianAccumulator {
  Callback callback = nullptr;
  void *user_data = nullptr;

  template <typename Params, typename Gradient, typename Hessian>
  Scalar operator()(const Params &params, Gradient &gradient, Hessian &hessian) const {
    Scalar cost = 0;
    Scalar *gradient_data = nullptr;
    if constexpr (!traits::is_nullptr_v<Gradient>) {
      gradient.setZero();
      gradient_data = gradient.data();
    }
    Handle handle = nullptr;
    if (hessian.rows() == params.dims()) handle = reinterpret_cast<Handle>(&hessian);
    if (callback(params.values.data(), static_cast<int>(params.dims()), &cost, gradient_data,
                 handle, user_data) != 0)
      throw UserStopRequested{};
    return cost;
  }
};

/// Runs a second-order optimizer with a preconfigured runtime Hessian layout.
template <typename Hessian, typename Params, typename Acc>
Summary OptimizeWithHessian(Params &params, const Acc &acc, const Options &options,
                            Hessian prototype) {
  using Scalar = typename Params::Scalar;
  prototype.setZero();
  switch (options.solver_type) {
    case Options::Solver::LevenbergMarquardt: {
      lm::Optimizer<Hessian> optimizer(options);
      optimizer.InitWith(Vector<Scalar, Dynamic>::Zero(params.dims()), std::move(prototype));
      return optimizer.Optimize(params, acc);
    }
    case Options::Solver::GaussNewton: {
      gn::Optimizer<Hessian> optimizer(options);
      optimizer.InitWith(Vector<Scalar, Dynamic>::Zero(params.dims()), std::move(prototype));
      return optimizer.Optimize(params, acc);
    }
    case Options::Solver::DogLeg: {
      dl::Optimizer<Hessian> optimizer(options);
      optimizer.InitWith(Vector<Scalar, Dynamic>::Zero(params.dims()), std::move(prototype));
      return optimizer.Optimize(params, acc);
    }
    default:
      throw std::invalid_argument(
          "Block Hessian entrypoints require Levenberg-Marquardt, Gauss-Newton or Dogleg");
  }
}

}  // namespace tinyopt::c_api_detail
