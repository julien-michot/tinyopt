// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tinyopt/optimizers/optimizer2.h>

namespace tinyopt::dl {

template <typename Hessian_t = MatX>
class Optimizer : public tinyopt::Optimizer2Base<Optimizer<Hessian_t>, Hessian_t> {
 public:
  using Base = tinyopt::Optimizer2Base<Optimizer<Hessian_t>, Hessian_t>;
  using Options = tinyopt::Options;

  struct State {
    using Scalar = typename Base::Scalar;

    Scalar radius = 1;
    mutable Scalar pending_prediction = 0;
    mutable Scalar pending_cost = 0;
    mutable bool pending_boundary = false;
    mutable bool pending_step = false;
  };

  explicit Optimizer(const Options &options = Options(Options::Solver::DogLeg))
      : Base(tinyopt::WithSolverOption(options, Options::Solver::DogLeg, "DogLeg")) {
    this->reset();
  }

  std::optional<typename Base::Grad_t> Solve() const override {
    return SolveDogLeg();
  }

  std::optional<typename Base::Grad_t> SolveDogLeg() const {
    using Scalar = typename Base::Scalar;
    using Grad_t = typename Base::Grad_t;
    if (!this->cost_.isValid()) return std::nullopt;
    const Scalar gradient_norm2 = this->grad_.squaredNorm();
    if (!std::isfinite(gradient_norm2)) return std::nullopt;
    if (gradient_norm2 <= std::numeric_limits<Scalar>::epsilon())
      return Grad_t::Zero(this->grad_.size());

    const auto gauss_newton = this->SolveLinear(-this->grad_);
    const auto hessian_gradient = (this->H_ * this->grad_).eval();
    const Scalar curvature = this->grad_.dot(hessian_gradient);
    const Grad_t cauchy =
        curvature > std::numeric_limits<Scalar>::epsilon()
            ? Grad_t(-(gradient_norm2 / curvature) * this->grad_)
            : Grad_t(-(state_.radius / std::sqrt(gradient_norm2)) * this->grad_);

    Grad_t step;
    bool on_boundary = false;
    if (gauss_newton && gauss_newton->norm() <= state_.radius) {
      step = *gauss_newton;
    } else if (!gauss_newton || cauchy.norm() >= state_.radius) {
      step = (state_.radius / cauchy.norm()) * cauchy;
      on_boundary = true;
    } else {
      const Grad_t difference = *gauss_newton - cauchy;
      const Scalar a = difference.squaredNorm();
      const Scalar b = Scalar(2) * cauchy.dot(difference);
      const Scalar c = cauchy.squaredNorm() - state_.radius * state_.radius;
      const Scalar discriminant = std::max<Scalar>(0, b * b - Scalar(4) * a * c);
      const Scalar tau = std::clamp((-b + std::sqrt(discriminant)) / (Scalar(2) * a),
                                    Scalar(0), Scalar(1));
      step = cauchy + tau * difference;
      on_boundary = true;
    }
    state_.pending_prediction = -Scalar(2) * this->grad_.dot(step) - step.dot(this->H_ * step);
    state_.pending_cost = this->cost_.cost;
    state_.pending_boundary = on_boundary;
    state_.pending_step =
        state_.pending_prediction > 0 && std::isfinite(state_.pending_prediction);
    return step;
  }

 protected:
  void ResetStrategy() override {
    state_.radius = std::max<typename Base::Scalar>(
        this->options_.dl.radius_init, std::numeric_limits<typename Base::Scalar>::epsilon());
    state_.pending_step = false;
  }
  void UpdateTrustRegion(typename Base::Scalar cost) override {
    if (!state_.pending_step) return;
    const auto ratio = (state_.pending_cost - cost) / state_.pending_prediction;
    const auto &dogleg = this->options_.dl;
    if (ratio < typename Base::Scalar(0.25))
      state_.radius *= dogleg.shrink_factor;
    else if (ratio > typename Base::Scalar(0.75) && state_.pending_boundary)
      state_.radius = std::min<typename Base::Scalar>(dogleg.radius_max,
                                                       state_.radius * dogleg.expand_factor);
    state_.pending_step = false;
  }

 private:
  State state_;

 public:
  void GoodStep(typename Base::Scalar) override {}
  void BadStep(typename Base::Scalar = 0) override {}
};

}  // namespace tinyopt::dl
