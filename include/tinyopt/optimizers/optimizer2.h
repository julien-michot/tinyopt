// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tinyopt/optimizers/optimizer.h>

namespace tinyopt {

template <typename Derived, typename Hessian_t>
class Optimizer2Base
    : public OptimizerCore<Derived, typename Hessian_t::Scalar,
                           SQRT(traits::params_trait<Hessian_t>::Dims), false, true> {
 public:
  using Scalar = typename Hessian_t::Scalar;
  static constexpr Index Dims = SQRT(traits::params_trait<Hessian_t>::Dims);
  using Grad_t = Vector<Scalar, Dims>;
  using H_t = Hessian_t;
  using Options = tinyopt::Options;
  using Base = OptimizerCore<Derived, Scalar, Dims, false, true>;

  explicit Optimizer2Base(const Options &options) : Base(NormalizeOptions(options)) {}
  virtual ~Optimizer2Base() = default;
  Optimizer2Base(const Optimizer2Base &) = default;
  Optimizer2Base &operator=(const Optimizer2Base &) = default;
  Optimizer2Base(Optimizer2Base &&) = default;
  Optimizer2Base &operator=(Optimizer2Base &&) = default;

  void InitWith(const Grad_t &gradient, const H_t &hessian) {
    grad_ = gradient;
    H_ = hessian;
  }

  void reset() {
    clear();
    ResetStrategy();
  }

  template <int D = Dims, std::enable_if_t<D == Dynamic, int> = 0>
  bool resize(int dims) {
    if (dims <= 0) throw std::invalid_argument("Dimensions must be positive");
    if (grad_.rows() == dims && H_.rows() == dims) return false;
    grad_.resize(dims);
    H_.resize(dims, dims);
    ResizeStrategy(dims);
    return true;
  }

  template <int D = Dims, std::enable_if_t<D != Dynamic, int> = 0>
  bool resize(int dims = Dims) {
    if (dims != Dims) throw std::invalid_argument("Static and dynamic dimensions must match");
    if constexpr (traits::is_sparse_matrix_v<H_t>) {
      grad_.resize(dims);
      H_.resize(dims, dims);
      ResizeStrategy(dims);
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
    if (ShouldRebuildLinearSystem()) {
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
      ApplyJacobiScaling();
    } else {
      Evaluate(x, acc, true);
      if (!cost_.isValid()) return false;
    }

    ApplyDamping();
#if defined(TINYOPT_ENABLE_SUITESPARSE)
    if constexpr (traits::is_sparse_matrix_v<H_t>) {
      if (this->options_.linear_solver == LinearSolverMethod::SuiteSparse)
        H_.makeCompressed();
    }
#endif
    UpdateTrustRegion(cost_.cost);
    return true;
  }

  virtual std::optional<Grad_t> Solve() const = 0;
  virtual void GoodStep(Scalar) = 0;
  virtual void BadStep(Scalar = 0) = 0;

  void FailedStep() { BadStep(); }
  virtual void Rebuild(bool rebuild) { RebuildStrategy(rebuild); }
  virtual std::string stateAsString() const { return {}; }
  Index dims() const { return grad_.size(); }
  const Cost &cost() const { return cost_; }
  const Grad_t &Gradient() const { return grad_; }
  Grad_t &Gradient() { return grad_; }
  Scalar GradientSquaredNorm() const { return grad_.squaredNorm(); }
  const H_t &H() const { return H_; }
  H_t &H() { return H_; }

  H_t Hessian() const {
    H_t hessian = H_;
    RestoreHessian(hessian);
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

 protected:
  static Options NormalizeOptions(const Options &options) {
    Options normalized = options;
    if (normalized.solver_type == Options::Solver::GradientDescent ||
        normalized.solver_type == Options::Solver::ConjugateGradient ||
        normalized.solver_type == Options::Solver::BFGS ||
        normalized.solver_type == Options::Solver::LBFGS)
      normalized.solver_type = Options::Solver::LevenbergMarquardt;
    return normalized;
  }

  virtual void ResetStrategy() = 0;
  virtual void ResizeStrategy(Index) {}
  virtual bool ShouldRebuildLinearSystem() const { return true; }
  virtual void ApplyJacobiScaling() {}
  virtual void ApplyDamping() {}
  virtual void UpdateTrustRegion(Scalar) {}
  virtual void RebuildStrategy(bool) {}
  virtual void RestoreHessian(H_t &) const {}

  Grad_t grad_;
  H_t H_;
  Cost cost_;

 protected:
  template <typename VectorType>
  std::optional<Grad_t> SolveLinear(const VectorType &rhs) const {
    return tinyopt::SolveLinearSystem(H_, rhs, this->options_.linear_solver,
                                      this->options_.svd_relative_threshold);
  }

 private:
  void NormalizeCost(Cost &value) const {
    if (!this->options_.cost.use_squared_norm) value.cost = std::sqrt(value.cost);
    if (this->options_.cost.downscale_by_2) value.cost *= 0.5;
    if (this->options_.cost.normalize && value.num_resisuals > 0)
      value.cost /= value.num_resisuals;
  }
};

}  // namespace tinyopt
