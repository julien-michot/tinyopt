// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tinyopt/optimizers/optimizer.h>

namespace tinyopt {

template <typename Hessian_t>
class Optimizer2
    : public OptimizerCore<Optimizer2<Hessian_t>, typename Hessian_t::Scalar,
                           SQRT(traits::params_trait<Hessian_t>::Dims), false, true> {
 public:
  using Scalar = typename Hessian_t::Scalar;
  static constexpr Index Dims = SQRT(traits::params_trait<Hessian_t>::Dims);
  using Grad_t = Vector<Scalar, Dims>;
  using H_t = Hessian_t;
  using Options = tinyopt::Options;
  using Base = OptimizerCore<Optimizer2<Hessian_t>, Scalar, Dims, false, true>;

  explicit Optimizer2(const Options &options = {}) : Base(NormalizeOptions(options)) {
    if (options.solver_type == Options::Solver::GradientDescent ||
        options.solver_type == Options::Solver::ConjugateGradient ||
        options.solver_type == Options::Solver::BFGS ||
        options.solver_type == Options::Solver::LBFGS)
      TINYOPT_LOG("⚠️ Second-order optimizer received a first-order solver; using LevenbergMarquardt");
    reset();
  }

  void InitWith(const Grad_t &gradient, const H_t &hessian) {
    grad_ = gradient;
    H_ = hessian;
  }

  void reset() {
    clear();
    lambda_ = this->options_.lm.damping_init;
    prev_lambda_ = 0;
    bad_factor_ = this->options_.lm.bad_factor;
    rebuild_linear_system_ = true;
    radius_ = std::max<Scalar>(this->options_.dl.radius_init,
                               std::numeric_limits<Scalar>::epsilon());
    pending_step_ = false;
  }

  template <int D = Dims, std::enable_if_t<D == Dynamic, int> = 0>
  bool resize(int dims) {
    if (dims <= 0) throw std::invalid_argument("Dimensions must be positive");
    if (grad_.rows() == dims && H_.rows() == dims) return false;
    grad_.resize(dims);
    H_.resize(dims, dims);
    if (this->options_.lm.jacobi_scaling) scaling_.resize(dims);
    return true;
  }

  template <int D = Dims, std::enable_if_t<D != Dynamic, int> = 0>
  bool resize(int dims = Dims) {
    if (dims != Dims) throw std::invalid_argument("Static and dynamic dimensions must match");
    if constexpr (traits::is_sparse_matrix_v<H_t>) {
      grad_.resize(dims);
      H_.resize(dims, dims);
      return true;
    }
    return false;
  }

  void clear() {
    grad_.setZero();
    H_.setZero();
  }

  template <typename X_t>
  bool ResizeIfNeeded(const X_t &x) {
    if constexpr (Dims == Dynamic) {
      const Index dims = traits::params_trait<X_t>::dims(x);
      if (grad_.rows() != dims) return resize(dims);
    }
    return false;
  }

  template <typename X_t, typename AccFunc>
  Scalar Evaluate(const X_t &x, const AccFunc &acc, bool save) {
    std::nullptr_t null_gradient;
    H_t null_hessian;
    Cost value = acc(x, null_gradient, null_hessian);
    NormalizeCost(value);
    if (save) cost_ = value;
    return value.cost;
  }

  template <typename X_t, typename AccFunc>
  bool Accumulate(const X_t &x, const AccFunc &acc) {
    cost_ = acc(x, grad_, H_);
    NormalizeCost(cost_);
    return cost_.isValid();
  }

  template <typename X_t, typename AccFunc>
  bool Build(const X_t &x, const AccFunc &acc, bool resize_and_clear = true) {
    const bool is_lm = this->options_.solver_type == Options::Solver::LevenbergMarquardt;
    if (!is_lm || rebuild_linear_system_) {
      if (resize_and_clear) {
        ResizeIfNeeded(x);
        clear();
      }
      if (!Accumulate(x, acc)) return false;
      this->Clamp(grad_, this->options_.opt.grad_clipping);
      if (this->options_.hessian.check_min_H_diag > 0 &&
          (H_.diagonal().cwiseAbs().array() <
           this->options_.hessian.check_min_H_diag).any())
        return false;
      if (!this->options_.hessian.H_is_full &&
          RequiresFullMatrix(this->options_.linear_solver))
        CompleteSymmetricMatrix(H_);
      if (is_lm && this->options_.lm.jacobi_scaling) ApplyJacobiScaling();
    } else {
      Evaluate(x, acc, true);
      if (!cost_.isValid()) return false;
    }

    if (is_lm && lambda_ > 0) {
      const Scalar factor = rebuild_linear_system_ ? Scalar(1) + lambda_
                                                   : (Scalar(1) + lambda_) /
                                                         (Scalar(1) + prev_lambda_);
      for (Index index = 0; index < H_.rows(); ++index) {
        if constexpr (traits::is_sparse_matrix_v<H_t>)
          H_.coeffRef(index, index) *= factor;
        else
          H_(index, index) *= factor;
      }
    }
#if defined(TINYOPT_ENABLE_SUITESPARSE)
    if constexpr (traits::is_sparse_matrix_v<H_t>) {
      if (this->options_.linear_solver == LinearSolverMethod::SuiteSparse)
        H_.makeCompressed();
    }
#endif
    if (this->options_.solver_type == Options::Solver::DogLeg && pending_step_)
      UpdateRadius(cost_.cost);
    return true;
  }

  std::optional<Grad_t> Solve() const {
    switch (this->options_.solver_type) {
      case Options::Solver::GaussNewton:
        return SolveGN();
      case Options::Solver::LevenbergMarquardt:
        return SolveLM();
      case Options::Solver::DogLeg:
        return SolveDogLeg();
      default:
        throw std::invalid_argument("Second-order optimizer received a first-order solver");
    }
  }

  std::optional<Grad_t> SolveGN() const {
    if (!cost_.isValid()) return std::nullopt;
    return SolveLinear(-grad_);
  }

  std::optional<Grad_t> SolveLM() const {
    auto step = SolveGN();
    if (step && this->options_.lm.jacobi_scaling)
      step->array() *= scaling_.array();
    return step;
  }

  std::optional<Grad_t> SolveDogLeg() const {
    if (!cost_.isValid()) return std::nullopt;
    const Scalar gradient_norm2 = grad_.squaredNorm();
    if (!std::isfinite(gradient_norm2)) return std::nullopt;
    if (gradient_norm2 <= std::numeric_limits<Scalar>::epsilon())
      return Grad_t::Zero(grad_.size());

    const auto gauss_newton = SolveLinear(-grad_);
    const auto hessian_gradient = (H_ * grad_).eval();
    const Scalar curvature = grad_.dot(hessian_gradient);
    const Grad_t cauchy =
        curvature > std::numeric_limits<Scalar>::epsilon()
            ? Grad_t(-(gradient_norm2 / curvature) * grad_)
            : Grad_t(-(radius_ / std::sqrt(gradient_norm2)) * grad_);

    Grad_t step;
    bool on_boundary = false;
    if (gauss_newton && gauss_newton->norm() <= radius_) {
      step = *gauss_newton;
    } else if (!gauss_newton || cauchy.norm() >= radius_) {
      step = (radius_ / cauchy.norm()) * cauchy;
      on_boundary = true;
    } else {
      const Grad_t difference = *gauss_newton - cauchy;
      const Scalar a = difference.squaredNorm();
      const Scalar b = Scalar(2) * cauchy.dot(difference);
      const Scalar c = cauchy.squaredNorm() - radius_ * radius_;
      const Scalar discriminant = std::max<Scalar>(0, b * b - Scalar(4) * a * c);
      const Scalar tau = std::clamp((-b + std::sqrt(discriminant)) / (Scalar(2) * a),
                                    Scalar(0), Scalar(1));
      step = cauchy + tau * difference;
      on_boundary = true;
    }
    pending_prediction_ = -Scalar(2) * grad_.dot(step) - step.dot(H_ * step);
    pending_cost_ = cost_.cost;
    pending_boundary_ = on_boundary;
    pending_step_ = pending_prediction_ > 0 && std::isfinite(pending_prediction_);
    return step;
  }

  void GoodStep(Scalar quality) {
    if (this->options_.solver_type != Options::Solver::LevenbergMarquardt) return;
    const auto &lm = this->options_.lm;
    Scalar factor = lm.good_factor;
    if (quality != Scalar(0))
      factor = std::max<Scalar>(factor, Scalar(1) - std::pow(Scalar(2) * quality - Scalar(1), 3));
    if (bad_factor_ != lm.bad_factor) factor /= bad_factor_;
    prev_lambda_ = lambda_;
    lambda_ = std::clamp<Scalar>(lambda_ * factor, lm.damping_range[0], lm.damping_range[1]);
    bad_factor_ = lm.bad_factor;
  }

  void BadStep(Scalar = 0) {
    if (this->options_.solver_type == Options::Solver::LevenbergMarquardt) {
      const auto &lm = this->options_.lm;
      prev_lambda_ = lambda_;
      lambda_ = std::clamp<Scalar>(lambda_ * bad_factor_, lm.damping_range[0], lm.damping_range[1]);
      bad_factor_ *= lm.bad_factor;
    }
  }

  void FailedStep() { BadStep(); }
  void Rebuild(bool rebuild) { rebuild_linear_system_ = rebuild; }
  std::string stateAsString() const {
    if (this->options_.solver_type != Options::Solver::LevenbergMarquardt) return {};
    std::ostringstream stream;
    stream << TINYOPT_FORMAT_NS::format("○:{:.2e} ", 1.0 / lambda_);
    return stream.str();
  }
  Index dims() const { return grad_.size(); }
  const Cost &cost() const { return cost_; }
  const Grad_t &Gradient() const { return grad_; }
  Grad_t &Gradient() { return grad_; }
  Scalar GradientSquaredNorm() const { return grad_.squaredNorm(); }
  const H_t &H() const { return H_; }
  H_t &H() { return H_; }
  H_t Hessian() const {
    H_t hessian = H_;
    if (this->options_.solver_type == Options::Solver::LevenbergMarquardt) {
      if (prev_lambda_ > 0) {
        const Scalar factor = Scalar(1) + prev_lambda_;
        for (Index index = 0; index < hessian.rows(); ++index) {
          if constexpr (traits::is_sparse_matrix_v<H_t>)
            hessian.coeffRef(index, index) /= factor;
          else
            hessian(index, index) /= factor;
        }
      }
      if (this->options_.lm.jacobi_scaling) ApplyJacobiScaling(hessian, true);
    }
    return hessian;
  }
  Scalar MaxStdDev() const {
    const auto covariance = InvCov(Hessian());
    if (!covariance) return 0;
    if constexpr (traits::is_sparse_matrix_v<H_t>)
      return std::sqrt(covariance->coeffs().maxCoeff());
    else
      return std::sqrt(covariance->maxCoeff());
  }

 private:
  static Options NormalizeOptions(const Options &options) {
    Options normalized = options;
    if (normalized.solver_type == Options::Solver::GradientDescent ||
        normalized.solver_type == Options::Solver::ConjugateGradient ||
        normalized.solver_type == Options::Solver::BFGS ||
        normalized.solver_type == Options::Solver::LBFGS)
      normalized.solver_type = Options::Solver::LevenbergMarquardt;
    return normalized;
  }

  template <typename VectorType>
  std::optional<Grad_t> SolveLinear(const VectorType &rhs) const {
    return tinyopt::SolveLinearSystem(H_, rhs, this->options_.linear_solver,
                                      this->options_.svd_relative_threshold);
  }

  void NormalizeCost(Cost &value) const {
    if (!this->options_.cost.use_squared_norm) value.cost = std::sqrt(value.cost);
    if (this->options_.cost.downscale_by_2) value.cost *= 0.5;
    if (this->options_.cost.normalize && value.num_resisuals > 0)
      value.cost /= value.num_resisuals;
  }

  void ApplyJacobiScaling() {
    constexpr Scalar min_diagonal = Scalar(1e-6);
    constexpr Scalar max_diagonal = Scalar(1e32);
    scaling_.resize(H_.rows());
    for (Index index = 0; index < H_.rows(); ++index) {
      const Scalar diagonal = std::clamp(H_.coeff(index, index), min_diagonal, max_diagonal);
      scaling_[index] = Scalar(1) / std::sqrt(diagonal);
    }
    grad_.array() *= scaling_.array();
    ApplyJacobiScaling(H_, false);
  }

  void ApplyJacobiScaling(H_t &hessian, bool inverse) const {
    const auto scale = [&](auto &value, Index row, Index column) {
      const Scalar factor = scaling_[row] * scaling_[column];
      if (inverse)
        value /= factor;
      else
        value *= factor;
    };
    if constexpr (traits::is_sparse_matrix_v<H_t>) {
      for (Index outer = 0; outer < hessian.outerSize(); ++outer)
        for (typename H_t::InnerIterator entry(hessian, outer); entry; ++entry)
          scale(entry.valueRef(), entry.row(), entry.col());
    } else {
      for (Index column = 0; column < hessian.cols(); ++column)
        for (Index row = 0; row < hessian.rows(); ++row)
          scale(hessian(row, column), row, column);
    }
  }

  void UpdateRadius(Scalar current_cost) {
    const Scalar ratio = (pending_cost_ - current_cost) / pending_prediction_;
    const auto &dogleg = this->options_.dl;
    if (ratio < Scalar(0.25))
      radius_ *= dogleg.shrink_factor;
    else if (ratio > Scalar(0.75) && pending_boundary_)
      radius_ = std::min<Scalar>(dogleg.radius_max, radius_ * dogleg.expand_factor);
    pending_step_ = false;
  }

  Grad_t grad_;
  H_t H_;
  Cost cost_;
  Grad_t scaling_;
  Scalar lambda_ = 1e-4;
  Scalar prev_lambda_ = 0;
  Scalar bad_factor_ = 2;
  Scalar radius_ = 1;
  mutable Scalar pending_prediction_ = 0;
  mutable Scalar pending_cost_ = 0;
  mutable bool pending_boundary_ = false;
  mutable bool pending_step_ = false;
  bool rebuild_linear_system_ = true;
};

}  // namespace tinyopt
