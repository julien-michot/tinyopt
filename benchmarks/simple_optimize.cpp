// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <tinyopt/optimize.h>

int main() {
  tinyopt::Vec2 fixed = tinyopt::Vec2::Ones();
  tinyopt::VecXf dynamic = tinyopt::VecXf::Ones(2);
  const auto loss = [](const auto &value) { return value.squaredNorm(); };

  (void)tinyopt::Optimize(fixed, loss);
  (void)tinyopt::Optimize(dynamic, loss);
}