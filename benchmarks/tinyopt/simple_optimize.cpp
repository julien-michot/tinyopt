// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <tinyopt/optimize.h>

int main() {
  tinyopt::Vec2 fixed = tinyopt::Vec2::Ones();
  tinyopt::VecXf dynamic = tinyopt::VecXf::Ones(2);
  const auto loss1 = [](const auto& value, auto& gradient, auto& hessian) {
    if constexpr (!tinyopt::traits::is_nullptr_v<decltype(gradient)>) {
      gradient = 2 * value;
      hessian.setIdentity();
      hessian *= 2;
    }
    return value.squaredNorm();
  };
  const auto loss2 = [](const auto& value, auto& gradient) {
    gradient = 2 * value;
    return value.squaredNorm();
  };
  tinyopt::Options fixed_options;
  tinyopt::lm::Optimizer<tinyopt::Mat2> fixed_optimizer(fixed_options);
  (void)fixed_optimizer(fixed, loss1);
  tinyopt::Options dynamic_options(tinyopt::Options::Solver::GradientDescent);
  tinyopt::gd::Optimizer<tinyopt::VecXf> dynamic_optimizer(dynamic_options);
  (void)dynamic_optimizer(dynamic, loss2);
}
