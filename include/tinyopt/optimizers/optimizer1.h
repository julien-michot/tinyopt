// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tinyopt/optimizers/optimizer.h>

namespace tinyopt {

template <typename Gradient_t>
class Optimizer1
    : public OptimizerCore<Optimizer1<Gradient_t>, typename Gradient_t::Scalar,
                           traits::params_trait<Gradient_t>::Dims, true, false> {
 public:
  using Scalar = typename Gradient_t::Scalar;
  static constexpr Index Dims = traits::params_trait<Gradient_t>::Dims;
  using Grad_t = Gradient_t;
  using Matrix_t = Matrix<Scalar, Dims, Dims>;
  using Options = tinyopt::Options;
  using Base = OptimizerCore<Optimizer1<Gradient_t>, Scalar, Dims, true, false>;
  static constexpr std::size_t HistoryCapacity = 8;

  explicit Optimizer1(
      const Options &options = Options(Options::Solver::GradientDescent))
      : Base(NormalizeOptions(options)),
        cg_step_size_(options.cg.step_size),
        bfgs_step_size_(options.bfgs.step_size),
        lbfgs_step_size_(options.lbfgs.step_size),
        lbfgs_history_limit_(std::clamp<std::size_t>(options.lbfgs.history_size, 1,
                                                     HistoryCapacity)) {
    reset();
  }

  void InitWith(const Grad_t &gradient) { grad_ = gradient; }

  void reset() {
    grad_.setZero();
    cost_ = Cost{};
    cg_step_size_ = this->options_.cg.step_size;
    bfgs_step_size_ = this->options_.bfgs.step_size;
    lbfgs_step_size_ = this->options_.lbfgs.step_size;
    has_cg_previous_ = false;
    bfgs_matrix_initialized_ = false;
    bfgs_has_gradient_ = false;
    bfgs_has_step_ = false;
    bfgs_update_available_ = false;
    lbfgs_history_count_ = 0;
    lbfgs_history_start_ = 0;
    lbfgs_has_gradient_ = false;
    lbfgs_has_step_ = false;
    lbfgs_update_available_ = false;
  }

  template <int D = Dims, std::enable_if_t<D == Dynamic, int> = 0>
  bool resize(int dims) {
    if (dims <= 0) throw std::invalid_argument("Dimensions must be positive");
    if (grad_.rows() == dims) return false;
    grad_.resize(dims);
    ResizeWorkspace();
    return true;
  }

  template <int D = Dims, std::enable_if_t<D != Dynamic, int> = 0>
  bool resize(int dims = Dims) {
    if (dims != Dims) throw std::invalid_argument("Static and dynamic dimensions must match");
    if constexpr (traits::is_sparse_matrix_v<Grad_t>) {
      grad_.resize(dims);
      return true;
    }
    return false;
  }

  template <typename X_t>
  bool ResizeIfNeeded(const X_t &x) {
    if constexpr (Dims == Dynamic) {
      const Index dims = traits::params_trait<X_t>::dims(x);
      if (grad_.rows() != dims) return resize(dims);
    }
    return false;
  }

  template <typename AccFunc>
  auto GetAccFunc(const AccFunc &acc) const {
    return [&](const auto &x, auto &gradient) -> Cost {
      using X_t = decltype(x);
      if constexpr (std::is_invocable_v<const AccFunc &, const X_t &, decltype(gradient),
                                        std::nullptr_t &>) {
        std::nullptr_t null_hessian;
        return acc(x, gradient, null_hessian);
      } else {
        return acc(x, gradient);
      }
    };
  }

  template <typename X_t, typename AccFunc>
  Scalar Evaluate(const X_t &x, const AccFunc &acc, bool save) {
    std::nullptr_t null_gradient;
    const auto adapted = GetAccFunc(acc);
    Cost value = adapted(x, null_gradient);
    NormalizeCost(value);
    if (save) cost_ = value;
    return value.cost;
  }

  template <typename X_t, typename AccFunc>
  bool Build(const X_t &x, const AccFunc &acc, bool resize_and_clear = true) {
    if constexpr (Dims == Dynamic) ResizeIfNeeded(x);
    if (!bfgs_matrix_initialized_) ResizeWorkspace();
    PrepareQuasiNewtonUpdate();
    if (resize_and_clear) grad_.setZero();
    const auto adapted = GetAccFunc(acc);
    cost_ = adapted(x, grad_);
    NormalizeCost(cost_);
    this->Clamp(grad_, this->options_.opt.grad_clipping);
    if (!cost_.isValid()) {
      bfgs_has_gradient_ = false;
      bfgs_update_available_ = false;
      lbfgs_has_gradient_ = false;
      lbfgs_update_available_ = false;
      return false;
    }

    switch (this->options_.solver_type) {
      case Options::Solver::ConjugateGradient:
        BuildConjugateGradient();
        break;
      case Options::Solver::BFGS:
        FinishBFGSBuild();
        break;
      case Options::Solver::LBFGS:
        FinishLBFGSBuild();
        break;
      default:
        break;
    }
    return true;
  }

  std::optional<Vector<Scalar, Dims>> Solve() const {
    if (!cost_.isValid()) return std::nullopt;
    switch (this->options_.solver_type) {
      case Options::Solver::GradientDescent:
        return -this->options_.gd.lr * grad_;
      case Options::Solver::ConjugateGradient:
        return (cg_step_size_ * cg_direction_).eval();
      case Options::Solver::BFGS:
        return SolveBFGS();
      case Options::Solver::LBFGS:
        return SolveLBFGS();
      default:
        throw std::invalid_argument("First-order optimizer received a second-order solver");
    }
  }

  void GoodStep(Scalar) {
    if (this->options_.solver_type == Options::Solver::BFGS) {
      UpdateBFGS();
      bfgs_update_available_ = false;
      bfgs_step_size_ =
          std::min<Scalar>(this->options_.bfgs.max_step_size,
                           bfgs_step_size_ * this->options_.bfgs.step_growth);
    } else if (this->options_.solver_type == Options::Solver::LBFGS) {
      UpdateLBFGS();
      lbfgs_update_available_ = false;
      lbfgs_step_size_ =
          std::min<Scalar>(this->options_.lbfgs.max_step_size,
                           lbfgs_step_size_ * this->options_.lbfgs.step_growth);
    }
  }

  void BadStep(Scalar = 0) {
    if (this->options_.solver_type == Options::Solver::ConjugateGradient) {
      cg_step_size_ *= this->options_.cg.step_reduction;
      has_cg_previous_ = false;
    } else if (this->options_.solver_type == Options::Solver::BFGS) {
      bfgs_step_size_ *= this->options_.bfgs.step_reduction;
      bfgs_has_gradient_ = false;
      bfgs_has_step_ = false;
      bfgs_update_available_ = false;
    } else if (this->options_.solver_type == Options::Solver::LBFGS) {
      lbfgs_step_size_ *= this->options_.lbfgs.step_reduction;
      lbfgs_has_gradient_ = false;
      lbfgs_has_step_ = false;
      lbfgs_update_available_ = false;
    }
  }

  void FailedStep() {}
  void Rebuild(bool) {}
  std::string stateAsString() const { return {}; }
  Index dims() const { return grad_.size(); }
  const Cost &cost() const { return cost_; }
  const Grad_t &Gradient() const { return grad_; }
  Grad_t &Gradient() { return grad_; }
  Scalar GradientSquaredNorm() const { return grad_.squaredNorm(); }

 private:
  static Options NormalizeOptions(const Options &options) {
    Options normalized = options;
    if (normalized.solver_type == Options::Solver::LevenbergMarquardt ||
        normalized.solver_type == Options::Solver::GaussNewton ||
        normalized.solver_type == Options::Solver::DogLeg) {
      TINYOPT_LOG("⚠️ First-order optimizer received a second-order solver; using GradientDescent");
      normalized.solver_type = Options::Solver::GradientDescent;
    }
    return normalized;
  }

  void NormalizeCost(Cost &value) const {
    if (!this->options_.cost.use_squared_norm) value.cost = std::sqrt(value.cost);
    if (this->options_.cost.downscale_by_2) value.cost *= 0.5;
    if (this->options_.cost.normalize && value.num_resisuals > 0)
      value.cost /= value.num_resisuals;
  }

  void ResizeWorkspace() {
    const Index dims = grad_.size();
    previous_gradient_.resize(dims);
    previous_direction_.resize(dims);
    cg_direction_.resize(dims);
    bfgs_inverse_hessian_.resize(dims, dims);
    bfgs_update_previous_gradient_.resize(dims);
    bfgs_update_step_.resize(dims);
    bfgs_gradient_delta_.resize(dims);
    bfgs_last_step_.resize(dims);
    for (auto &step : lbfgs_steps_) step.resize(dims);
    for (auto &gradient : lbfgs_gradients_) gradient.resize(dims);
    lbfgs_update_previous_gradient_.resize(dims);
    lbfgs_update_step_.resize(dims);
    lbfgs_gradient_delta_.resize(dims);
    lbfgs_last_step_.resize(dims);
    previous_gradient_.setZero();
    previous_direction_.setZero();
    bfgs_inverse_hessian_.setIdentity();
    bfgs_matrix_initialized_ = true;
    lbfgs_history_count_ = 0;
    lbfgs_history_start_ = 0;
    has_cg_previous_ = false;
    bfgs_has_gradient_ = false;
    bfgs_has_step_ = false;
    lbfgs_has_gradient_ = false;
    lbfgs_has_step_ = false;
  }

  void PrepareQuasiNewtonUpdate() {
    const Index dims = grad_.size();
    if constexpr (Dims == Dynamic) {
      if (!bfgs_matrix_initialized_ || bfgs_inverse_hessian_.rows() != dims) ResizeWorkspace();
    }
    if (this->options_.solver_type == Options::Solver::BFGS && bfgs_has_gradient_ &&
        bfgs_has_step_) {
      bfgs_update_step_ = bfgs_last_step_;
      bfgs_update_previous_gradient_ = grad_;
    }
    if (this->options_.solver_type == Options::Solver::LBFGS && lbfgs_has_gradient_ &&
        lbfgs_has_step_) {
      lbfgs_update_step_ = lbfgs_last_step_;
      lbfgs_update_previous_gradient_ = grad_;
    }
  }

  void BuildConjugateGradient() {
    Scalar beta = 0;
    if (has_cg_previous_) {
      const Scalar denominator = previous_gradient_.squaredNorm();
      if (denominator > std::numeric_limits<Scalar>::epsilon())
        beta = std::max<Scalar>(
            0, grad_.dot(grad_ - previous_gradient_) / denominator);
    }
    cg_direction_ = -grad_ + beta * previous_direction_;
    if (grad_.dot(cg_direction_) >= 0) cg_direction_ = -grad_;
    previous_gradient_ = grad_;
    previous_direction_ = cg_direction_;
    has_cg_previous_ = true;
  }

  void FinishBFGSBuild() {
    bfgs_update_available_ = bfgs_has_gradient_ && bfgs_has_step_;
    if (bfgs_update_available_) bfgs_gradient_delta_ = grad_ - bfgs_update_previous_gradient_;
    bfgs_has_gradient_ = true;
    bfgs_has_step_ = false;
  }

  void FinishLBFGSBuild() {
    lbfgs_update_available_ = lbfgs_has_gradient_ && lbfgs_has_step_;
    if (lbfgs_update_available_)
      lbfgs_gradient_delta_ = grad_ - lbfgs_update_previous_gradient_;
    lbfgs_has_gradient_ = true;
    lbfgs_has_step_ = false;
  }

  std::optional<Vector<Scalar, Dims>> SolveBFGS() const {
    Vector<Scalar, Dims> direction = -(bfgs_inverse_hessian_ * grad_).eval();
    if (!direction.allFinite() || grad_.dot(direction) >= 0) {
      bfgs_inverse_hessian_.setIdentity();
      direction = -grad_;
    }
    bfgs_last_step_ = bfgs_step_size_ * direction;
    bfgs_has_step_ = true;
    return bfgs_last_step_;
  }

  std::size_t LBFGSIndex(std::size_t offset) const {
    return (lbfgs_history_start_ + offset) % HistoryCapacity;
  }

  std::optional<Vector<Scalar, Dims>> SolveLBFGS() const {
    Vector<Scalar, Dims> q = grad_;
    for (std::size_t offset = 0; offset < lbfgs_history_count_; ++offset) {
      const std::size_t index = LBFGSIndex(lbfgs_history_count_ - offset - 1);
      lbfgs_alpha_[index] = lbfgs_rho_[index] * lbfgs_steps_[index].dot(q);
      q.noalias() -= lbfgs_alpha_[index] * lbfgs_gradients_[index];
    }
    Scalar scale = 1;
    if (lbfgs_history_count_ > 0) {
      const std::size_t newest = LBFGSIndex(lbfgs_history_count_ - 1);
      const Scalar yy = lbfgs_gradients_[newest].squaredNorm();
      if (yy > std::numeric_limits<Scalar>::epsilon())
        scale = lbfgs_steps_[newest].dot(lbfgs_gradients_[newest]) / yy;
    }
    Vector<Scalar, Dims> direction = -scale * q;
    for (std::size_t offset = 0; offset < lbfgs_history_count_; ++offset) {
      const std::size_t index = LBFGSIndex(offset);
      const Scalar beta = lbfgs_rho_[index] * lbfgs_gradients_[index].dot(direction);
      direction.noalias() +=
          (lbfgs_alpha_[index] - beta) * lbfgs_steps_[index];
    }
    if (!direction.allFinite() || grad_.dot(direction) >= 0) {
      lbfgs_history_count_ = 0;
      lbfgs_history_start_ = 0;
      direction = -grad_;
    }
    lbfgs_last_step_ = lbfgs_step_size_ * direction;
    lbfgs_has_step_ = true;
    return lbfgs_last_step_;
  }

  void UpdateBFGS() {
    if (!bfgs_update_available_) return;
    const Scalar curvature = bfgs_update_step_.dot(bfgs_gradient_delta_);
    const Scalar threshold = this->options_.bfgs.curvature_threshold *
                             bfgs_update_step_.norm() * bfgs_gradient_delta_.norm();
    if (std::isfinite(curvature) && curvature > threshold) {
      const Scalar rho = Scalar(1) / curvature;
      Matrix_t transform = Matrix_t::Identity(bfgs_inverse_hessian_.rows(),
                                               bfgs_inverse_hessian_.cols());
      transform.noalias() -= rho * bfgs_update_step_ * bfgs_gradient_delta_.transpose();
      Matrix_t updated = (transform * bfgs_inverse_hessian_ * transform.transpose()).eval();
      updated.noalias() += rho * bfgs_update_step_ * bfgs_update_step_.transpose();
      bfgs_inverse_hessian_.swap(updated);
    }
  }

  void UpdateLBFGS() {
    if (!lbfgs_update_available_) return;
    const Scalar curvature = lbfgs_update_step_.dot(lbfgs_gradient_delta_);
    const Scalar threshold = this->options_.lbfgs.curvature_threshold *
                             lbfgs_update_step_.norm() * lbfgs_gradient_delta_.norm();
    if (!std::isfinite(curvature) || curvature <= threshold) return;
    std::size_t index;
    if (lbfgs_history_count_ < lbfgs_history_limit_) {
      index = LBFGSIndex(lbfgs_history_count_++);
    } else {
      index = lbfgs_history_start_;
      lbfgs_history_start_ = (lbfgs_history_start_ + 1) % HistoryCapacity;
    }
    lbfgs_steps_[index] = lbfgs_update_step_;
    lbfgs_gradients_[index] = lbfgs_gradient_delta_;
    lbfgs_rho_[index] = Scalar(1) / curvature;
  }

  Grad_t grad_;
  Cost cost_;
  Vector<Scalar, Dims> previous_gradient_;
  Vector<Scalar, Dims> previous_direction_;
  Vector<Scalar, Dims> cg_direction_;
  mutable Matrix_t bfgs_inverse_hessian_;
  Vector<Scalar, Dims> bfgs_update_previous_gradient_;
  Vector<Scalar, Dims> bfgs_update_step_;
  Vector<Scalar, Dims> bfgs_gradient_delta_;
  mutable Vector<Scalar, Dims> bfgs_last_step_;
  std::array<Vector<Scalar, Dims>, HistoryCapacity> lbfgs_steps_;
  std::array<Vector<Scalar, Dims>, HistoryCapacity> lbfgs_gradients_;
  mutable std::array<Scalar, HistoryCapacity> lbfgs_alpha_{};
  std::array<Scalar, HistoryCapacity> lbfgs_rho_{};
  Vector<Scalar, Dims> lbfgs_update_previous_gradient_;
  Vector<Scalar, Dims> lbfgs_update_step_;
  Vector<Scalar, Dims> lbfgs_gradient_delta_;
  mutable Vector<Scalar, Dims> lbfgs_last_step_;
  Scalar cg_step_size_;
  Scalar bfgs_step_size_;
  Scalar lbfgs_step_size_;
  std::size_t lbfgs_history_limit_;
  mutable std::size_t lbfgs_history_count_ = 0;
  mutable std::size_t lbfgs_history_start_ = 0;
  bool has_cg_previous_ = false;
  bool bfgs_matrix_initialized_ = false;
  bool bfgs_has_gradient_ = false;
  mutable bool bfgs_has_step_ = false;
  bool bfgs_update_available_ = false;
  bool lbfgs_has_gradient_ = false;
  mutable bool lbfgs_has_step_ = false;
  bool lbfgs_update_available_ = false;
};

}  // namespace tinyopt
