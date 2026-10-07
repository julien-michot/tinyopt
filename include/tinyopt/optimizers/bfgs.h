// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tinyopt/optimizers/lbfgs.h>
#include <tinyopt/optimizers/optimizer1.h>

namespace tinyopt::bfgs {

template <typename Scalar, Index Dims>
struct State {
  using Grad_t = Vector<Scalar, Dims>;
  using Matrix_t = Matrix<Scalar, Dims, Dims>;

  mutable Matrix_t inverse_hessian;
  Grad_t update_previous_gradient;
  Grad_t update_step;
  Grad_t gradient_delta;
  mutable Grad_t last_step;
  Scalar step_size = 0;
  bool matrix_initialized = false;
  bool has_gradient = false;
  mutable bool has_step = false;
  bool update_available = false;
};

template <typename Gradient_t = VecX>
class Optimizer : public tinyopt::Optimizer1Base<Optimizer<Gradient_t>, Gradient_t> {
 public:
  using Base = tinyopt::Optimizer1Base<Optimizer<Gradient_t>, Gradient_t>;
  using Options = tinyopt::Options;
  using StrategyState = bfgs::State<typename Base::Scalar, Base::Dims>;

  explicit Optimizer(const Options &options = Options(Options::Solver::BFGS))
      : Base(tinyopt::WithSolverOption(options, Options::Solver::BFGS, "BFGS")) {
    this->reset();
  }

  std::optional<typename Base::Grad_t> Solve() const override {
    if (!this->cost_.isValid()) return std::nullopt;
    typename Base::Grad_t direction = -(state_.inverse_hessian * this->grad_).eval();
    if (!direction.allFinite() || this->grad_.dot(direction) >= 0) {
      state_.inverse_hessian.setIdentity();
      direction = -this->grad_;
    }
    state_.last_step = state_.step_size * direction;
    state_.has_step = true;
    return state_.last_step;
  }

 protected:
  void ResetStrategy() override {
    state_.step_size = this->options_.bfgs.step_size;
    state_.matrix_initialized = false;
    state_.has_gradient = false;
    state_.has_step = false;
    state_.update_available = false;
  }
  void ResizeStrategy(tinyopt::Index dims) override {
    state_.inverse_hessian.resize(dims, dims);
    state_.update_previous_gradient.resize(dims);
    state_.update_step.resize(dims);
    state_.gradient_delta.resize(dims);
    state_.last_step.resize(dims);
    state_.inverse_hessian.setIdentity();
    state_.matrix_initialized = true;
    state_.has_gradient = false;
    state_.has_step = false;
  }
  void PrepareStrategyBuild() override {
    if constexpr (Base::Dims == tinyopt::Dynamic) {
      if (!state_.matrix_initialized || state_.inverse_hessian.rows() != this->grad_.size())
        ResizeStrategy(this->grad_.size());
    } else if (!state_.matrix_initialized) {
      ResizeStrategy(this->grad_.size());
    }
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
  StrategyState state_;

 public:
  void GoodStep(typename Base::Scalar) override {
    Update();
    state_.update_available = false;
    state_.step_size = std::min<typename Base::Scalar>(
        this->options_.bfgs.max_step_size,
        state_.step_size * this->options_.bfgs.step_growth);
  }
  void BadStep(typename Base::Scalar = 0) override {
    state_.step_size *= this->options_.bfgs.step_reduction;
    state_.has_gradient = false;
    state_.has_step = false;
    state_.update_available = false;
  }

 private:
  void Update() {
    if (!state_.update_available) return;
    const auto curvature = state_.update_step.dot(state_.gradient_delta);
    const auto threshold = this->options_.bfgs.curvature_threshold *
                           state_.update_step.norm() * state_.gradient_delta.norm();
    if (std::isfinite(curvature) && curvature > threshold) {
      using Matrix_t = typename Base::Matrix_t;
      using Scalar = typename Base::Scalar;
      Matrix_t transform =
          Matrix_t::Identity(state_.inverse_hessian.rows(), state_.inverse_hessian.cols());
      transform.noalias() -=
          (Scalar(1) / curvature) * state_.update_step * state_.gradient_delta.transpose();
      Matrix_t updated =
          (transform * state_.inverse_hessian * transform.transpose()).eval();
      updated.noalias() += (Scalar(1) / curvature) * state_.update_step *
                           state_.update_step.transpose();
      state_.inverse_hessian.swap(updated);
    }
  }
};

}  // namespace tinyopt::bfgs
