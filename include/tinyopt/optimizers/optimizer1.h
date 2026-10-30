// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tinyopt/optimizers/optimizer.h>

namespace tinyopt {

template <typename Derived, typename Gradient_t>
class Optimizer1Base : public OptimizerCore<Derived, typename Gradient_t::Scalar,
                                            traits::params_trait<Gradient_t>::Dims, true, false> {
 public:
  using Scalar = typename Gradient_t::Scalar;
  static constexpr Index Dims = traits::params_trait<Gradient_t>::Dims;
  using Grad_t = Gradient_t;
  using Matrix_t = Matrix<Scalar, Dims, Dims>;
  using Options = tinyopt::Options;
  using Base = OptimizerCore<Derived, Scalar, Dims, true, false>;
  explicit Optimizer1Base(const Options &options) : Base(options) {}
  virtual ~Optimizer1Base() = default;
  Optimizer1Base(const Optimizer1Base &) = default;
  Optimizer1Base &operator=(const Optimizer1Base &) = default;
  Optimizer1Base(Optimizer1Base &&) = default;
  Optimizer1Base &operator=(Optimizer1Base &&) = default;

  void InitWith(const Grad_t &gradient) {
    this->dims_ = gradient.size();
    grad_ = gradient;
    grad_squared_norm_ = grad_.squaredNorm();
    grad_max_norm_ = grad_.template lpNorm<Eigen::Infinity>();
  }

  void clear() {
    grad_.setZero();
    grad_squared_norm_ = 0;
    grad_max_norm_ = 0;
  }

  void reset() {
    clear();
    cost_ = Cost{};
    ResetStrategy();
  }

  template <int D = Dims, std::enable_if_t<D == Dynamic, int> = 0>
  bool resize(int dims) {
    if (dims <= 0) throw std::invalid_argument("Dimensions must be positive");
    this->dims_ = dims;
    if (grad_.rows() == dims) return false;
    grad_.resize(dims);
    ResizeStrategy(dims);
    return true;
  }

  template <int D = Dims, std::enable_if_t<D != Dynamic, int> = 0>
  bool resize(int dims = Dims) {
    if (dims != Dims) throw std::invalid_argument("Static and dynamic dimensions must match");
    if constexpr (traits::is_sparse_matrix_v<Grad_t>) {
      grad_.resize(dims);
      ResizeStrategy(dims);
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

  template <typename X_t, typename AccFunc>
  Scalar Evaluate(const X_t &x, const AccFunc &acc, bool save) {
    Cost value = EvaluateFunc(x, acc);
    NormalizeCost(value);
    if (save) cost_ = value;
    return value.cost;
  }

  template <typename X_t, typename AccFunc>
  Scalar EvaluateAt(X_t &x, const Vector<Scalar, Dims> &dx, const AccFunc &acc) {
    using ptrait = traits::params_trait<X_t>;
    ptrait::PlusEq(x, dx);
    Cost value = Evaluate(x, acc, false);
    ptrait::PlusEq(x, (-dx).eval());
    return value.cost;
  }

  template <typename X_t, typename AccFunc>
  bool Build(const X_t &x, const AccFunc &acc, bool resize_and_clear = true) {
    if constexpr (Dims == Dynamic) ResizeIfNeeded(x);
    PrepareStrategyBuild();
    if (resize_and_clear) grad_.setZero();
    cost_ = AccumulateFunc(x, grad_, acc);
    NormalizeCost(cost_);
    this->Clamp(grad_, this->options_.opt.grad_clipping);
    ApplyFixedStates();
    grad_squared_norm_ = grad_.squaredNorm();
    grad_max_norm_ = grad_.template lpNorm<Eigen::Infinity>();
    if (!cost_.isValid()) {
      InvalidateStrategyBuild();
      return false;
    }
    BuildStrategy();
    return true;
  }

 private:
  template <typename X_t, typename AccFunc>
  Cost EvaluateFunc(const X_t &x, const AccFunc &acc) const {
    if constexpr (std::is_invocable_v<const AccFunc &, const X_t &>) {
      return acc(x);
    } else if constexpr (std::is_invocable_v<const AccFunc &, const X_t &, std::nullptr_t &>) {
      std::nullptr_t null_gradient;
      return acc(x, null_gradient);
    } else if constexpr (std::is_invocable_v<const AccFunc &, const X_t &, std::nullptr_t &,
                                             std::nullptr_t &>) {
      std::nullptr_t null_gradient;
      std::nullptr_t null_hessian;
      return acc(x, null_gradient, null_hessian);
    } else {
      static_assert(
          std::is_invocable_v<const AccFunc &, const X_t &> ||
              std::is_invocable_v<const AccFunc &, const X_t &, std::nullptr_t &> ||
              std::is_invocable_v<const AccFunc &, const X_t &, std::nullptr_t &, std::nullptr_t &>,
          "Cost or accumulation function must support evaluation without gradient: either as a "
          "cost-only function 'acc(x)' or by accepting std::nullptr_t for the gradient (e.g. "
          "'if constexpr (!traits::is_nullptr_v<decltype(grad)>)').");
    }
  }

  template <typename X_t, typename GradType, typename AccFunc>
  Cost AccumulateFunc(const X_t &x, GradType &gradient, const AccFunc &acc) const {
    if constexpr (std::is_invocable_v<const AccFunc &, const X_t &, GradType &, std::nullptr_t &>) {
      std::nullptr_t null_hessian;
      return acc(x, gradient, null_hessian);
    } else if constexpr (std::is_invocable_v<const AccFunc &, const X_t &, GradType &>) {
      return acc(x, gradient);
    } else {
      return acc(x);
    }
  }

 public:
  void ApplyFixedStates() {
    if (!this->hasFixed()) return;
    const auto apply_zero = [&](Index idx) {
      if (idx >= 0 && idx < grad_.size()) {
        if constexpr (traits::is_sparse_matrix_v<Grad_t>) {
          grad_.coeffRef(idx, 0) = Scalar(0);
        } else {
          grad_(idx) = Scalar(0);
        }
      }
    };
    if constexpr (Dims == Dynamic) {
      for (auto idx : this->fixed_indices_) apply_zero(idx);
    } else {
      for (Index i = 0; i < Dims; ++i) {
        if (this->fixed_indices_[i]) apply_zero(i);
      }
    }
  }

  virtual std::optional<Vector<Scalar, Dims>> Solve() const = 0;
  virtual void GoodStep(Scalar) = 0;
  virtual void BadStep(Scalar = 0) = 0;
  void FailedStep() {}
  void Rebuild(bool) {}
  std::string stateAsString() const { return {}; }
  Index dims() const { return grad_.size() > 0 ? grad_.size() : this->dims_; }
  const Cost &cost() const { return cost_; }
  const Grad_t &Gradient() const { return grad_; }
  Grad_t &Gradient() { return grad_; }
  Scalar GradientSquaredNorm() const { return grad_squared_norm_; }
  Scalar GradientMaxNorm() const { return grad_max_norm_; }
  [[nodiscard]] bool hasReduced() const { return false; }
  [[nodiscard]] Index numReduced() const { return 0; }
  Scalar PredictedReduction() const { return 0; }

 protected:
  virtual void ResetStrategy() = 0;
  virtual void ResizeStrategy(Index) {}
  virtual void PrepareStrategyBuild() {}
  virtual void InvalidateStrategyBuild() {}
  virtual void BuildStrategy() = 0;

  void NormalizeCost(Cost &value) const {
    if (!this->options_.cost.use_squared_norm) value.cost = std::sqrt(value.cost);
    if (this->options_.cost.downscale_by_2) value.cost *= 0.5;
    if (this->options_.cost.normalize && value.num_resisuals > 0) value.cost /= value.num_resisuals;
  }

  Grad_t grad_;
  Cost cost_;
  Scalar grad_squared_norm_ = 0;
  Scalar grad_max_norm_ = 0;
};

}  // namespace tinyopt
