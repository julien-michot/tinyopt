// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tinyopt/optimizers/optimizer2.h>

namespace tinyopt::gn {

template <typename Hessian_t = MatX>
class Optimizer : public tinyopt::Optimizer2Base<Optimizer<Hessian_t>, Hessian_t> {
 public:
  using Base = tinyopt::Optimizer2Base<Optimizer<Hessian_t>, Hessian_t>;
  using Options = tinyopt::Options;

  explicit Optimizer(const Options &options = Options(Options::Solver::GaussNewton))
      : Base(tinyopt::WithSolverOption(options, Options::Solver::GaussNewton, "Gauss-Newton")) {
    this->reset();
  }

  std::optional<typename Base::Grad_t> Solve() const override { return SolveGN(); }
  std::optional<typename Base::Grad_t> SolveGN() const {
    if (!this->cost_.isValid()) return std::nullopt;
    return this->SolveLinear(-this->grad_);
  }

 protected:
  void ResetStrategy() override {}

 public:
  void GoodStep(typename Base::Scalar) override {}
  void BadStep(typename Base::Scalar = 0) override {}
};

}  // namespace tinyopt::gn
