// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

#include <tinyopt/solvers/gn.h>

namespace tinyopt::solvers {

template <typename Hessian_t = MatX>
class SolverDogLeg : public SolverGN<Hessian_t> {
 public:
  using Base = SolverGN<Hessian_t>;
  using Scalar = typename Base::Scalar;
  using Grad_t = typename Base::Grad_t;
  using H_t = typename Base::H_t;
  static constexpr Index Dims = Base::Dims;
  static constexpr bool IsNLLS = true;
  static constexpr bool FirstOrder = false;

  explicit SolverDogLeg(const Options &options = {})
      : Base(options), radius_(std::max<Scalar>(options.dl.radius_init,
                                                std::numeric_limits<Scalar>::epsilon())) {}

  void reset() override {
    Base::reset();
    radius_ = std::max<Scalar>(this->options_.dl.radius_init,
                               std::numeric_limits<Scalar>::epsilon());
    pending_step_ = false;
  }

  template <typename X_t, typename AccFunc>
  bool Build(const X_t &x, const AccFunc &acc, bool resize_and_clear = true) {
    if (!Base::Build(x, acc, resize_and_clear)) return false;
    if (pending_step_) UpdateRadius(this->cost().cost);
    return true;
  }

  std::optional<Vector<Scalar, Dims>> Solve() const override {
    if (!this->cost().isValid()) return std::nullopt;

    const auto &gradient = this->Gradient();
    const Scalar gradient_norm2 = gradient.squaredNorm();
    if (!std::isfinite(gradient_norm2)) return std::nullopt;
    if (gradient_norm2 <= std::numeric_limits<Scalar>::epsilon()) {
      pending_step_ = false;
      return Vector<Scalar, Dims>::Zero(gradient.size());
    }

    const auto gauss_newton =
        tinyopt::SolveLinearSystem(this->Hessian(), -gradient, this->options_.linear_solver);
    const auto hessian_gradient = (this->Hessian() * gradient).eval();
    const Scalar curvature = gradient.dot(hessian_gradient);
    Vector<Scalar, Dims> cauchy;
    if (curvature > std::numeric_limits<Scalar>::epsilon())
      cauchy = -(gradient_norm2 / curvature) * gradient;
    else
      cauchy = -(radius_ / std::sqrt(gradient_norm2)) * gradient;

    Vector<Scalar, Dims> step;
    bool on_boundary = false;
    if (gauss_newton && gauss_newton->norm() <= radius_) {
      step = *gauss_newton;
    } else if (cauchy.norm() >= radius_ || !gauss_newton) {
      const Scalar cauchy_norm = cauchy.norm();
      step = (radius_ / cauchy_norm) * cauchy;
      on_boundary = true;
    } else {
      const Vector<Scalar, Dims> difference = *gauss_newton - cauchy;
      const Scalar a = difference.squaredNorm();
      const Scalar b = Scalar(2) * cauchy.dot(difference);
      const Scalar c = cauchy.squaredNorm() - radius_ * radius_;
      const Scalar discriminant = std::max<Scalar>(0, b * b - Scalar(4) * a * c);
      const Scalar tau =
          std::clamp((-b + std::sqrt(discriminant)) / (Scalar(2) * a), Scalar(0), Scalar(1));
      step = cauchy + tau * difference;
      on_boundary = true;
    }

    const auto hessian_step = (this->Hessian() * step).eval();
    pending_prediction_ = -Scalar(2) * gradient.dot(step) - step.dot(hessian_step);
    pending_cost_ = this->cost().cost;
    pending_boundary_ = on_boundary;
    pending_step_ = pending_prediction_ > 0 && std::isfinite(pending_prediction_);
    return step;
  }

  Scalar TrustRegionRadius() const { return radius_; }

 private:
  void UpdateRadius(Scalar current_cost) {
    const Scalar ratio = (pending_cost_ - current_cost) / pending_prediction_;
    if (ratio < Scalar(0.25))
      radius_ *= this->options_.dl.shrink_factor;
    else if (ratio > Scalar(0.75) && pending_boundary_)
      radius_ = std::min<Scalar>(this->options_.dl.radius_max,
                                 radius_ * this->options_.dl.expand_factor);
    pending_step_ = false;
  }

  Scalar radius_;
  mutable Scalar pending_prediction_ = 0;
  mutable Scalar pending_cost_ = 0;
  mutable bool pending_boundary_ = false;
  mutable bool pending_step_ = false;
};

}  // namespace tinyopt::solvers