// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <limits>

#include <tinyopt/solvers/gd.h>

namespace tinyopt::solvers {

template <typename Gradient_t = VecX>
class SolverCG : public SolverGD<Gradient_t> {
 public:
  using Base = SolverGD<Gradient_t>;
  using Scalar = typename Base::Scalar;
  using Grad_t = typename Base::Grad_t;
  static constexpr Index Dims = Base::Dims;
  static constexpr bool IsNLLS = false;
  static constexpr bool FirstOrder = true;

  explicit SolverCG(const Options &options = {}) : Base(options), step_size_(options.cg.step_size) {
    // `Grad_t::Zero()` is invalid for dynamic sizes, so zero-initialize here instead
    ResetDirection();
    direction_.setZero();
  }

  void reset() {
    Base::reset();
    ResetDirection();
    step_size_ = this->options_.cg.step_size;
  }

  template <typename X_t, typename AccFunc>
  bool Build(const X_t &x, const AccFunc &acc, bool resize_and_clear = true) {
    if (!Base::Build(x, acc, resize_and_clear)) return false;
    ResizeWorkspace();

    Scalar beta = 0;
    if (has_previous_) {
      const Scalar denominator = previous_gradient_.squaredNorm();
      if (denominator > std::numeric_limits<Scalar>::epsilon()) {
        beta = std::max<Scalar>(0, this->grad_.dot(this->grad_ - previous_gradient_) / denominator);
      }
    }

    direction_ = -this->grad_ + beta * previous_direction_;
    if (this->grad_.dot(direction_) >= 0) direction_ = -this->grad_;
    previous_gradient_ = this->grad_;
    previous_direction_ = direction_;
    has_previous_ = true;
    return true;
  }

  std::optional<Vector<Scalar, Dims>> Solve() const override {
    if (!this->cost().isValid()) return std::nullopt;
    return (step_size_ * direction_).eval();
  }

  void BadStep(Scalar /*quality*/ = 0) override {
    step_size_ *= this->options_.cg.step_reduction;
    ResetDirection();
  }

 private:
  void ResizeWorkspace() {
    if (previous_gradient_.size() == this->grad_.size()) return;
    previous_gradient_.resize(this->grad_.size());
    previous_direction_.resize(this->grad_.size());
    previous_gradient_.setZero();
    previous_direction_.setZero();
    has_previous_ = false;
  }

  void ResetDirection() {
    previous_gradient_.setZero();
    previous_direction_.setZero();
    has_previous_ = false;
  }

  Grad_t previous_gradient_;
  Grad_t previous_direction_;
  Grad_t direction_;
  Scalar step_size_;
  bool has_previous_ = false;
};

}  // namespace tinyopt::solvers