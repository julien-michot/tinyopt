// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tinyopt/optimizers/optimizer.h>

namespace tinyopt {

template <typename Derived, typename Gradient_t>
class Optimizer1Base
    : public OptimizerCore<Derived, typename Gradient_t::Scalar,
                           traits::params_trait<Gradient_t>::Dims, true, false> {
 public:
  using Scalar = typename Gradient_t::Scalar;
  static constexpr Index Dims = traits::params_trait<Gradient_t>::Dims;
  using Grad_t = Gradient_t;
  using Matrix_t = Matrix<Scalar, Dims, Dims>;
  using Options = tinyopt::Options;
  using Base = OptimizerCore<Derived, Scalar, Dims, true, false>;
  explicit Optimizer1Base(const Options &options) : Base(NormalizeOptions(options)) {}
  virtual ~Optimizer1Base() = default;
  Optimizer1Base(const Optimizer1Base &) = default;
  Optimizer1Base &operator=(const Optimizer1Base &) = default;
  Optimizer1Base(Optimizer1Base &&) = default;
  Optimizer1Base &operator=(Optimizer1Base &&) = default;

  void InitWith(const Grad_t &gradient) { grad_ = gradient; }

  void reset() {
    grad_.setZero();
    cost_ = Cost{};
    ResetStrategy();
  }

  template <int D = Dims, std::enable_if_t<D == Dynamic, int> = 0>
  bool resize(int dims) {
    if (dims <= 0) throw std::invalid_argument("Dimensions must be positive");
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
    PrepareStrategyBuild();
    if (resize_and_clear) grad_.setZero();
    const auto adapted = GetAccFunc(acc);
    cost_ = adapted(x, grad_);
    NormalizeCost(cost_);
    this->Clamp(grad_, this->options_.opt.grad_clipping);
    if (!cost_.isValid()) {
      InvalidateStrategyBuild();
      return false;
    }
    BuildStrategy();
    return true;
  }

  virtual std::optional<Vector<Scalar, Dims>> Solve() const = 0;
  virtual void GoodStep(Scalar) = 0;
  virtual void BadStep(Scalar = 0) = 0;
  void FailedStep() {}
  void Rebuild(bool) {}
  std::string stateAsString() const { return {}; }
  Index dims() const { return grad_.size(); }
  const Cost &cost() const { return cost_; }
  const Grad_t &Gradient() const { return grad_; }
  Grad_t &Gradient() { return grad_; }
  Scalar GradientSquaredNorm() const { return grad_.squaredNorm(); }

 protected:
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

  virtual void ResetStrategy() = 0;
  virtual void ResizeStrategy(Index) {}
  virtual void PrepareStrategyBuild() {}
  virtual void InvalidateStrategyBuild() {}
  virtual void BuildStrategy() = 0;

  void NormalizeCost(Cost &value) const {
    if (!this->options_.cost.use_squared_norm) value.cost = std::sqrt(value.cost);
    if (this->options_.cost.downscale_by_2) value.cost *= 0.5;
    if (this->options_.cost.normalize && value.num_resisuals > 0)
      value.cost /= value.num_resisuals;
  }

  Grad_t grad_;
  Cost cost_;
};

}  // namespace tinyopt
