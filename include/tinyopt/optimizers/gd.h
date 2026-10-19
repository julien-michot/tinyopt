// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tinyopt/optimizers/optimizer1.h>

namespace tinyopt::gd {

template <typename Gradient_t = VecX>
class Optimizer : public tinyopt::Optimizer1Base<Optimizer<Gradient_t>, Gradient_t> {
 public:
  using Base = tinyopt::Optimizer1Base<Optimizer<Gradient_t>, Gradient_t>;
  using Options = tinyopt::Options;

  explicit Optimizer(const Options &options = Options(Options::Solver::GradientDescent))
      : Base(options) {
    if (options.solver_type != Options::Solver::GradientDescent) {
      TINYOPT_LOG("⚠ Optimizer received a different solver type, using GradientDescent");
      this->options_.solver_type = Options::Solver::GradientDescent;
    }
    this->reset();
  }

  std::optional<typename Base::Grad_t> Solve() const override {
    if (!this->cost_.isValid()) return std::nullopt;
    return -this->options_.gd.lr * this->grad_;
  }

 protected:
  void ResetStrategy() override {}
  void BuildStrategy() override {}

 public:
  void GoodStep(typename Base::Scalar) override {}
  void BadStep(typename Base::Scalar = 0) override {}
};

}  // namespace tinyopt::gd
