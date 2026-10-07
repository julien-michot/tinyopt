// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tinyopt/optimizers/optimizer1.h>

namespace tinyopt::gd {

template <typename Gradient_t = VecX>
class Optimizer : public tinyopt::Optimizer1<Gradient_t> {
 public:
  using Base = tinyopt::Optimizer1<Gradient_t>;
  using Options = tinyopt::Options;

  explicit Optimizer(const Options &options = Options(Options::Solver::GradientDescent))
      : Base(tinyopt::WithSolverOption(options, Options::Solver::GradientDescent,
                                       "Gradient-descent")) {}
};

}  // namespace tinyopt::gd
