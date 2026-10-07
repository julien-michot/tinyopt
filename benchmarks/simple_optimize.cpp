// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <tinyopt/optimize.h>

int main() {
  tinyopt::Vec2 fixed = tinyopt::Vec2::Ones();
  tinyopt::VecXf dynamic = tinyopt::VecXf::Ones(2);
  const auto loss1 = [](const auto &value) { return value.squaredNorm(); };
  const auto loss2 = [](const auto &value, auto &gradient) {
    gradient = 2 * value;
    return value.squaredNorm();
  };
  tinyopt::Options options;
  (void)tinyopt::Optimize(fixed, loss1, options);
  options.solver_type = tinyopt::Options::Solver::GradientDescent;
  (void)tinyopt::Optimize(dynamic, loss2, options);
}