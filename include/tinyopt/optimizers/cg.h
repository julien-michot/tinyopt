// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tinyopt/optimizers/optimizer1.h>

namespace tinyopt::cg {

template <typename Gradient_t = VecX>
class Optimizer : public tinyopt::Optimizer1Base<Optimizer<Gradient_t>, Gradient_t> {
 public:
  using Base = tinyopt::Optimizer1Base<Optimizer<Gradient_t>, Gradient_t>;
  using Options = tinyopt::Options;

  struct State {
    using Scalar = typename Base::Scalar;
    using Grad_t = typename Base::Grad_t;

    Grad_t previous_gradient;
    Grad_t previous_direction;
    Grad_t direction;
    Scalar step_size = 0;
    bool has_previous = false;
  };

  explicit Optimizer(const Options &options = Options(Options::Solver::ConjugateGradient))
      : Base(tinyopt::WithSolverOption(options, Options::Solver::ConjugateGradient,
                                       "Conjugate-gradient")) {
    this->reset();
  }

  std::optional<typename Base::Grad_t> Solve() const override {
    if (!this->cost_.isValid()) return std::nullopt;
    return (state_.step_size * state_.direction).eval();
  }

 protected:
  void ResetStrategy() override {
    state_.step_size = this->options_.cg.step_size;
    state_.has_previous = false;
  }
  void ResizeStrategy(tinyopt::Index dims) override {
    state_.previous_gradient.resize(dims);
    state_.previous_direction.resize(dims);
    state_.direction.resize(dims);
    state_.previous_gradient.setZero();
    state_.previous_direction.setZero();
    state_.has_previous = false;
  }
  void BuildStrategy() override {
    if (state_.previous_gradient.size() != this->grad_.size())
      ResizeStrategy(this->grad_.size());
    typename Base::Scalar beta = 0;
    if (state_.has_previous) {
      const auto denominator = state_.previous_gradient.squaredNorm();
      if (denominator > std::numeric_limits<typename Base::Scalar>::epsilon())
        beta = std::max<typename Base::Scalar>(
            0, this->grad_.dot(this->grad_ - state_.previous_gradient) / denominator);
    }
    state_.direction = -this->grad_ + beta * state_.previous_direction;
    if (this->grad_.dot(state_.direction) >= 0) state_.direction = -this->grad_;
    state_.previous_gradient = this->grad_;
    state_.previous_direction = state_.direction;
    state_.has_previous = true;
  }

 private:
  State state_;

 public:
  void GoodStep(typename Base::Scalar) override {}
  void BadStep(typename Base::Scalar = 0) override {
    state_.step_size *= this->options_.cg.step_reduction;
    state_.has_previous = false;
  }
};

}  // namespace tinyopt::cg
