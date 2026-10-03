// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#include <tinyopt/solvers/gd.h>

namespace tinyopt::solvers {

template <typename Gradient_t = VecX>
class SolverBFGS : public SolverGD<Gradient_t> {
 public:
  using Base = SolverGD<Gradient_t>;
  using Scalar = typename Base::Scalar;
  using Grad_t = typename Base::Grad_t;
  using Matrix_t = Matrix<Scalar, Base::Dims, Base::Dims>;
  static constexpr Index Dims = Base::Dims;
  static constexpr bool IsNLLS = false;
  static constexpr bool FirstOrder = true;

  explicit SolverBFGS(const Options &options = {}) : Base(options), step_size_(options.bfgs.step_size) {}

  void reset() {
    Base::reset();
    step_size_ = this->options_.bfgs.step_size;
    matrix_initialized_ = false;
    has_gradient_ = false;
    has_step_ = false;
    update_available_ = false;
  }

  template <typename X_t, typename AccFunc>
  bool Build(const X_t &x, const AccFunc &acc, bool resize_and_clear = true) {
    if constexpr (Dims == Dynamic) this->ResizeIfNeeded(x);
    InitializeWorkspace();

    const bool can_update = has_gradient_ && has_step_;
    if (can_update) {
      update_step_ = last_step_;
      update_previous_gradient_ = this->grad_;
    }

    if (!Base::Build(x, acc, resize_and_clear)) {
      has_gradient_ = false;
      update_available_ = false;
      return false;
    }

    update_available_ = can_update;
    if (can_update) update_gradient_delta_ = this->grad_ - update_previous_gradient_;
    has_gradient_ = true;
    has_step_ = false;
    return true;
  }

  std::optional<Vector<Scalar, Dims>> Solve() const override {
    if (!this->cost().isValid()) return std::nullopt;

    Vector<Scalar, Dims> direction = -(inverse_hessian_ * this->grad_).eval();
    if (!direction.allFinite() || this->grad_.dot(direction) >= 0) {
      inverse_hessian_.setIdentity();
      direction = -this->grad_;
    }

    last_step_ = step_size_ * direction;
    has_step_ = true;
    return last_step_;
  }

  void GoodStep(Scalar /*quality*/ = 0) override {
    if (update_available_) {
      const Scalar curvature = update_step_.dot(update_gradient_delta_);
      const Scalar minimum_curvature = this->options_.bfgs.curvature_threshold *
                                      update_step_.norm() * update_gradient_delta_.norm();
      if (std::isfinite(curvature) && curvature > minimum_curvature) {
        const Scalar rho = Scalar(1) / curvature;
        Matrix_t transform = Matrix_t::Identity(inverse_hessian_.rows(), inverse_hessian_.cols());
        transform.noalias() -= rho * update_step_ * update_gradient_delta_.transpose();
        Matrix_t updated = (transform * inverse_hessian_ * transform.transpose()).eval();
        updated.noalias() += rho * update_step_ * update_step_.transpose();
        inverse_hessian_.swap(updated);
      }
    }
    update_available_ = false;
    step_size_ = std::min<Scalar>(this->options_.bfgs.max_step_size,
                                  step_size_ * this->options_.bfgs.step_growth);
  }

  void BadStep(Scalar /*quality*/ = 0) override {
    step_size_ *= this->options_.bfgs.step_reduction;
    has_gradient_ = false;
    has_step_ = false;
    update_available_ = false;
  }

 private:
  void InitializeWorkspace() {
    const Index dims = this->grad_.size();
    if (!matrix_initialized_ || inverse_hessian_.rows() != dims) {
      inverse_hessian_.resize(dims, dims);
      inverse_hessian_.setIdentity();
      update_previous_gradient_.resize(dims);
      update_step_.resize(dims);
      update_gradient_delta_.resize(dims);
      last_step_.resize(dims);
      matrix_initialized_ = true;
      has_gradient_ = false;
      has_step_ = false;
    }
  }

  mutable Matrix_t inverse_hessian_;
  Vector<Scalar, Dims> update_previous_gradient_;
  Vector<Scalar, Dims> update_step_;
  Vector<Scalar, Dims> update_gradient_delta_;
  mutable Vector<Scalar, Dims> last_step_;
  Scalar step_size_;
  bool matrix_initialized_ = false;
  bool has_gradient_ = false;
  mutable bool has_step_ = false;
  bool update_available_ = false;
};

template <typename Gradient_t = VecX, std::size_t HistoryCapacity = 8>
class SolverLBFGS : public SolverGD<Gradient_t> {
 public:
  using Base = SolverGD<Gradient_t>;
  using Scalar = typename Base::Scalar;
  using Grad_t = typename Base::Grad_t;
  static constexpr Index Dims = Base::Dims;
  static constexpr bool IsNLLS = false;
  static constexpr bool FirstOrder = true;

  explicit SolverLBFGS(const Options &options = {})
      : Base(options),
        step_size_(options.lbfgs.step_size),
        history_limit_(std::clamp<std::size_t>(options.lbfgs.history_size, 1, HistoryCapacity)) {}

  void reset() {
    Base::reset();
    step_size_ = this->options_.lbfgs.step_size;
    history_count_ = 0;
    history_start_ = 0;
    has_gradient_ = false;
    has_step_ = false;
    update_available_ = false;
  }

  template <typename X_t, typename AccFunc>
  bool Build(const X_t &x, const AccFunc &acc, bool resize_and_clear = true) {
    if constexpr (Dims == Dynamic) this->ResizeIfNeeded(x);
    ResizeWorkspace();

    const bool can_update = has_gradient_ && has_step_;
    if (can_update) {
      update_step_ = last_step_;
      update_previous_gradient_ = this->grad_;
    }

    if (!Base::Build(x, acc, resize_and_clear)) {
      has_gradient_ = false;
      update_available_ = false;
      return false;
    }

    update_available_ = can_update;
    if (can_update) update_gradient_delta_ = this->grad_ - update_previous_gradient_;
    has_gradient_ = true;
    has_step_ = false;
    return true;
  }

  std::optional<Vector<Scalar, Dims>> Solve() const override {
    if (!this->cost().isValid()) return std::nullopt;

    Vector<Scalar, Dims> q = this->grad_;
    for (std::size_t offset = 0; offset < history_count_; ++offset) {
      const std::size_t index = HistoryIndex(history_count_ - offset - 1);
      alpha_[index] = rho_[index] * step_history_[index].dot(q);
      q.noalias() -= alpha_[index] * gradient_history_[index];
    }

    Scalar scale = 1;
    if (history_count_ > 0) {
      const std::size_t newest = HistoryIndex(history_count_ - 1);
      const Scalar yy = gradient_history_[newest].squaredNorm();
      if (yy > std::numeric_limits<Scalar>::epsilon())
        scale = step_history_[newest].dot(gradient_history_[newest]) / yy;
    }
    Vector<Scalar, Dims> direction = -scale * q;

    for (std::size_t offset = 0; offset < history_count_; ++offset) {
      const std::size_t index = HistoryIndex(offset);
      const Scalar beta = rho_[index] * gradient_history_[index].dot(direction);
      direction.noalias() += (alpha_[index] - beta) * step_history_[index];
    }

    if (!direction.allFinite() || this->grad_.dot(direction) >= 0) {
      history_count_ = 0;
      history_start_ = 0;
      direction = -this->grad_;
    }
    last_step_ = step_size_ * direction;
    has_step_ = true;
    return last_step_;
  }

  void GoodStep(Scalar /*quality*/ = 0) override {
    if (update_available_) {
      const Scalar curvature = update_step_.dot(update_gradient_delta_);
      const Scalar minimum_curvature = this->options_.lbfgs.curvature_threshold *
                                      update_step_.norm() * update_gradient_delta_.norm();
      if (std::isfinite(curvature) && curvature > minimum_curvature) StorePair(curvature);
    }
    update_available_ = false;
    step_size_ = std::min<Scalar>(this->options_.lbfgs.max_step_size,
                                  step_size_ * this->options_.lbfgs.step_growth);
  }

  void BadStep(Scalar /*quality*/ = 0) override {
    step_size_ *= this->options_.lbfgs.step_reduction;
    has_gradient_ = false;
    has_step_ = false;
    update_available_ = false;
  }

 private:
  std::size_t HistoryIndex(std::size_t offset) const {
    return (history_start_ + offset) % HistoryCapacity;
  }

  void ResizeWorkspace() {
    const Index dims = this->grad_.size();
    if (workspace_dims_ == dims) return;
    for (auto &step : step_history_) step.resize(dims);
    for (auto &gradient : gradient_history_) gradient.resize(dims);
    update_previous_gradient_.resize(dims);
    update_step_.resize(dims);
    update_gradient_delta_.resize(dims);
    last_step_.resize(dims);
    workspace_dims_ = dims;
    history_count_ = 0;
    history_start_ = 0;
    has_gradient_ = false;
    has_step_ = false;
  }

  void StorePair(Scalar curvature) {
    std::size_t index;
    if (history_count_ < history_limit_) {
      index = HistoryIndex(history_count_);
      ++history_count_;
    } else {
      index = history_start_;
      history_start_ = (history_start_ + 1) % HistoryCapacity;
    }
    step_history_[index] = update_step_;
    gradient_history_[index] = update_gradient_delta_;
    rho_[index] = Scalar(1) / curvature;
  }

  std::array<Vector<Scalar, Dims>, HistoryCapacity> step_history_;
  std::array<Vector<Scalar, Dims>, HistoryCapacity> gradient_history_;
  mutable std::array<Scalar, HistoryCapacity> alpha_{};
  std::array<Scalar, HistoryCapacity> rho_{};
  Vector<Scalar, Dims> update_previous_gradient_;
  Vector<Scalar, Dims> update_step_;
  Vector<Scalar, Dims> update_gradient_delta_;
  mutable Vector<Scalar, Dims> last_step_;
  Scalar step_size_;
  std::size_t history_limit_;
  mutable std::size_t history_count_ = 0;
  mutable std::size_t history_start_ = 0;
  Index workspace_dims_ = 0;
  bool has_gradient_ = false;
  mutable bool has_step_ = false;
  bool update_available_ = false;
};

}  // namespace tinyopt::solvers