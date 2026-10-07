// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tinyopt/optimizers/optimizer2.h>

namespace tinyopt::gn {

template <typename Hessian_t = MatX>
class Optimizer : public tinyopt::Optimizer2<Hessian_t> {
 public:
  using Base = tinyopt::Optimizer2<Hessian_t>;
  using Options = tinyopt::Options;

  explicit Optimizer(const Options &options = Options(Options::Solver::GaussNewton))
      : Base(tinyopt::WithSolverOption(options, Options::Solver::GaussNewton, "Gauss-Newton")) {}
};

}  // namespace tinyopt::gn
