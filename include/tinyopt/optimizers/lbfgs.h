// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tinyopt/optimizers/optimizer1.h>

namespace tinyopt::lbfgs {

template <typename Gradient_t = VecX>
class Optimizer : public tinyopt::Optimizer1Base<Optimizer<Gradient_t>, Gradient_t> {
 public:
  using Base = tinyopt::Optimizer1Base<Optimizer<Gradient_t>, Gradient_t>;
  using Options = tinyopt::Options;

  struct State {
    static constexpr std::size_t HistoryCapacity = 8;
    using Scalar = typename Base::Scalar;
    using Grad_t = typename Base::Grad_t;

    std::array<Grad_t, HistoryCapacity> steps;
    std::array<Grad_t, HistoryCapacity> gradients;
    mutable std::array<Scalar, HistoryCapacity> alpha{};
    std::array<Scalar, HistoryCapacity> rho{};
    Grad_t update_previous_gradient;
    Grad_t update_step;
    Grad_t gradient_delta;
    mutable Grad_t last_step;
    Scalar step_size = 0;
    std::size_t history_limit = HistoryCapacity;
    mutable std::size_t history_count = 0;
    mutable std::size_t history_start = 0;
    bool has_gradient = false;
    mutable bool has_step = false;
    bool update_available = false;
  };

  explicit Optimizer(const Options &options = Options(Options::Solver::LBFGS))
      : Base(tinyopt::WithSolverOption(options, Options::Solver::LBFGS, "L-BFGS")) {
    this->reset();
  }

  std::optional<typename Base::Grad_t> Solve() const override {
    if (!this->cost_.isValid()) return std::nullopt;
    typename Base::Grad_t q = this->grad_;
    for (std::size_t offset = 0; offset < state_.history_count; ++offset) {
      const std::size_t index = Index(state_.history_count - offset - 1);
      state_.alpha[index] = state_.rho[index] * state_.steps[index].dot(q);
      q.noalias() -= state_.alpha[index] * state_.gradients[index];
    }
    typename Base::Scalar scale = 1;
    if (state_.history_count > 0) {
      const std::size_t newest = Index(state_.history_count - 1);
      const auto yy = state_.gradients[newest].squaredNorm();
      if (yy > std::numeric_limits<typename Base::Scalar>::epsilon())
        scale = state_.steps[newest].dot(state_.gradients[newest]) / yy;
    }
    typename Base::Grad_t direction = -scale * q;
    for (std::size_t offset = 0; offset < state_.history_count; ++offset) {
      const std::size_t index = Index(offset);
      const auto beta = state_.rho[index] * state_.gradients[index].dot(direction);
      direction.noalias() += (state_.alpha[index] - beta) * state_.steps[index];
    }
    if (!direction.allFinite() || this->grad_.dot(direction) >= 0) {
      state_.history_count = 0;
      state_.history_start = 0;
      direction = -this->grad_;
    }
    state_.last_step = state_.step_size * direction;
    state_.has_step = true;
    return state_.last_step;
  }

 protected:
  void ResetStrategy() override {
    state_.step_size = this->options_.lbfgs.step_size;
    state_.history_limit = std::clamp<std::size_t>(this->options_.lbfgs.history_size, 1,
                                                   State::HistoryCapacity);
    state_.history_count = 0;
    state_.history_start = 0;
    state_.has_gradient = false;
    state_.has_step = false;
    state_.update_available = false;
  }
  void ResizeStrategy(tinyopt::Index dims) override {
    for (auto &step : state_.steps) step.resize(dims);
    for (auto &gradient : state_.gradients) gradient.resize(dims);
    state_.update_previous_gradient.resize(dims);
    state_.update_step.resize(dims);
    state_.gradient_delta.resize(dims);
    state_.last_step.resize(dims);
    state_.history_count = 0;
    state_.history_start = 0;
    state_.has_gradient = false;
    state_.has_step = false;
  }
  void PrepareStrategyBuild() override {
    if (state_.steps[0].size() != this->grad_.size()) ResizeStrategy(this->grad_.size());
    if (state_.has_gradient && state_.has_step) {
      state_.update_step = state_.last_step;
      state_.update_previous_gradient = this->grad_;
    }
  }
  void InvalidateStrategyBuild() override {
    state_.has_gradient = false;
    state_.update_available = false;
  }
  void BuildStrategy() override {
    state_.update_available = state_.has_gradient && state_.has_step;
    if (state_.update_available)
      state_.gradient_delta = this->grad_ - state_.update_previous_gradient;
    state_.has_gradient = true;
    state_.has_step = false;
  }

 private:
  State state_;

 public:
  void GoodStep(typename Base::Scalar) override {
    Update();
    state_.update_available = false;
    state_.step_size = std::min<typename Base::Scalar>(
        this->options_.lbfgs.max_step_size, state_.step_size * this->options_.lbfgs.step_growth);
  }
  void BadStep(typename Base::Scalar = 0) override {
    state_.step_size *= this->options_.lbfgs.step_reduction;
    state_.has_gradient = false;
    state_.has_step = false;
    state_.update_available = false;
  }

 private:
  static std::size_t Index(const State &state, std::size_t offset) {
    return (state.history_start + offset) % State::HistoryCapacity;
  }

  std::size_t Index(std::size_t offset) const {
    return Index(state_, offset);
  }

  void Update() {
    if (!state_.update_available) return;
    const auto curvature = state_.update_step.dot(state_.gradient_delta);
    const auto threshold = this->options_.lbfgs.curvature_threshold *
                           state_.update_step.norm() * state_.gradient_delta.norm();
    if (!std::isfinite(curvature) || curvature <= threshold) return;
    std::size_t index;
    if (state_.history_count < state_.history_limit) {
      index = Index(state_.history_count++);
    } else {
      index = state_.history_start;
      state_.history_start = (state_.history_start + 1) % State::HistoryCapacity;
    }
    state_.steps[index] = state_.update_step;
    state_.gradients[index] = state_.gradient_delta;
    state_.rho[index] = typename Base::Scalar(1) / curvature;
  }
};

}  // namespace tinyopt::lbfgs
