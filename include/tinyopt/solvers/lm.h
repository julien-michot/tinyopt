// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tinyopt/cost.h>
#include <tinyopt/log.h>
#include <tinyopt/solvers/gn.h>

namespace tinyopt::lm {

/***
 *  @brief Levenberg-Marquardt Solver Optimization options
 *
 ***/
using SolverOptions = tinyopt::Options;
}  // namespace tinyopt::lm

namespace tinyopt::solvers {

template <typename Hessian_t = MatX>
class SolverLM : public tinyopt::solvers::SolverGN<Hessian_t> {
 public:
  static constexpr bool IsNLLS = true;
  static constexpr bool FirstOrder = false;  // this is a pseudo second order algorithm
  using Base = tinyopt::solvers::SolverGN<Hessian_t>;
  using Scalar = typename Hessian_t::Scalar;
  static constexpr Index Dims = Base::Dims;

  // Hessian Type
  using H_t = Hessian_t;
  // Gradient Type
  using Grad_t = Vector<Scalar, Dims>;
  // Options
  using Options = tinyopt::Options;

  explicit SolverLM(const Options &options = {}) : Base(options), options_{options} { reset(); }

  /// Reset the solver state and clear gradient & hessian
  void reset() override {
    this->clear();
    lambda_ = options_.lm.damping_init;
    prev_lambda_ = 0;
    bad_factor_ = options_.lm.bad_factor;
    rebuild_linear_system_ = true;
  }

  /// Force the solver to rebuild or skip it
  void Rebuild(bool b) override { rebuild_linear_system_ = b; }

  /// Build the gradient and hessian by accumulating residuals and their jacobians
  /// Returns true on success
  template <typename X_t, typename AccFunc>
  inline bool Build(const X_t &x, const AccFunc &acc_func, bool resize_and_clear = true) {
    if (rebuild_linear_system_) {
      // Resize the system if needed and clear gradient
      if (resize_and_clear) {
        this->ResizeIfNeeded(x);
        this->clear();
      }

      // Accumulate residuals and update both gardient and Hessian approx (Jt*J)
      const bool success = this->Accumulate(x, acc_func);

      // Early skip on failure (no residuals)
      if (!success) {
        if (options_.log.enable)
          TINYOPT_LOG("❌ Failed to accumulate residuals: {}", this->cost().toString());
        return false;
      }

      // Eventually clip the gradient
      this->Clamp(this->grad_, options_.opt.grad_clipping);

      // Verify Hessian's diagonal
      if (options_.hessian.check_min_H_diag > 0 &&
          (this->H_.diagonal().cwiseAbs().array() < options_.hessian.check_min_H_diag).any()) {
        if (options_.log.enable) TINYOPT_LOG("❌ Hessian has very low diagonal coefficients");
        return false;
      }

      // Fill the lower part if H if needed
      if (!options_.hessian.H_is_full && RequiresFullMatrix(options_.linear_solver))
        CompleteSymmetricMatrix(this->H_);

      if (options_.lm.jacobi_scaling) ApplyJacobiScaling();

    } else {  // Keeping H and gradient, only evaluate the cost again

      this->Evaluate(x, acc_func, true);
      const bool success = this->cost().isValid();
      // Early skip on failure (no residuals)
      if (!success) {
        if (options_.log.enable) TINYOPT_LOG("❌ Failed to accumulate residuals");
        return false;
      }
    }

    // Damping the diagonal: d' = d + lambda*d
    if (lambda_ > 0.0) {
      const double s =
          rebuild_linear_system_ ? 1.0 + lambda_ : (1.0 + lambda_) / (1.0 + prev_lambda_);
      for (int i = 0; i < this->H_.rows(); ++i) {
        if constexpr (traits::is_matrix_or_array_v<H_t>)
          this->H_(i, i) *= s;
        else
          this->H_.coeffRef(i, i) *= s;
      }
    }

#if defined(TINYOPT_ENABLE_SUITESPARSE)
    if constexpr (traits::is_sparse_matrix_v<H_t>) {
      if (options_.linear_solver == LinearSolverMethod::SuiteSparse) this->H_.makeCompressed();
    }
#endif

    return true;
  }

  /// Damping stategy for a good step: increase the damping factor \lambda
  void GoodStep(Scalar quality) override {
    Scalar s = options_.lm.good_factor;  // Scale to apply on damping lambda

    // Use an approximative scaling based on the step quality TODO: improve this
    if (quality != Scalar(0.0)) {
      s = std::max<Scalar>(s, 1.0f - std::pow(2.0f * quality - 1.0f, 3.0f));
    }

    // Check whether the previous 'bad' step was actually good and revert the last scaling
    if (bad_factor_ != options_.lm.bad_factor) s /= bad_factor_;

    prev_lambda_ = lambda_;
    lambda_ =
        std::clamp<Scalar>(lambda_ * s, options_.lm.damping_range[0], options_.lm.damping_range[1]);
    bad_factor_ = options_.lm.bad_factor;
  }

  /// Damping stategy for a bad step: decrease the damping factor \lambda
  void BadStep(Scalar /*quality*/ = 0.0f) override {
    Scalar s = bad_factor_;  // Scale to apply on damping lambda
    prev_lambda_ = lambda_;
    lambda_ =
        std::clamp<Scalar>(lambda_ * s, options_.lm.damping_range[0], options_.lm.damping_range[1]);
    bad_factor_ *= options_.lm.bad_factor;
  }

  /// Damping stategy for a failure to solve the linear system, decrease the damping factor \lambda
  void FailedStep() override { BadStep(); }

  std::optional<Grad_t> Solve() const override {
    auto delta = Base::Solve();
    if (delta && options_.lm.jacobi_scaling) delta->array() *= scaling_.array();
    return delta;
  }

  std::string stateAsString() const override {
    std::ostringstream oss;
    oss << TINYOPT_FORMAT_NS::format("○:{:.2e} ", 1.0 / lambda_);
    return oss.str();
  }

  /// Latest Hessian approximation (JtJ), un-damped
  H_t Hessian() const {
    H_t H = this->H_;
    if (prev_lambda_ > 0.0) {
      const Scalar s = 1.0f + prev_lambda_;
      for (int i = 0; i < this->H_.cols(); ++i) {
        if constexpr (traits::is_matrix_or_array_v<H_t>)
          H(i, i) /= s;
        else
          H.coeffRef(i, i) = this->H_.coeff(i, i) / s;
      }
    }
    if (options_.lm.jacobi_scaling) ApplyJacobiScaling(H, true);
    return H;
  }

  /// Latest Covariance estimate
  std::optional<H_t> Covariance() const override { return InvCov(Hessian()); }

  /// Return the square root of the maximum (co)variance of the H.inv()
  /// H being the damped Hessian H_ if use_damped == true (faster) or un-damped Hessian() (accurate)
  Scalar MaxStdDev(bool use_damped = true) const {
    H_t H = use_damped ? this->H_ : Hessian();
    if (use_damped && options_.lm.jacobi_scaling) ApplyJacobiScaling(H, true);
    const auto I = InvCov(H);
    if (!I) return 0;
    using std::sqrt;
    if constexpr (traits::is_sparse_matrix_v<H_t>)
      return sqrt(I.value().coeffs().maxCoeff());
    else
      return sqrt(I.value().maxCoeff());
  }

 protected:
  void ApplyJacobiScaling() {
    constexpr Scalar MinDiagonal = Scalar(1e-6);
    constexpr Scalar MaxDiagonal = Scalar(1e32);
    scaling_.resize(this->H_.rows());
    for (Index index = 0; index < this->H_.rows(); ++index) {
      const Scalar diagonal = std::clamp(this->H_.coeff(index, index), MinDiagonal, MaxDiagonal);
      scaling_[index] = Scalar(1) / std::sqrt(diagonal);
    }
    this->grad_.array() *= scaling_.array();
    ApplyJacobiScaling(this->H_, false);
  }

  void ApplyJacobiScaling(H_t &hessian, bool inverse) const {
    const auto scale_entry = [&](auto &value, Index row, Index column) {
      const Scalar factor = scaling_[row] * scaling_[column];
      if (inverse)
        value /= factor;
      else
        value *= factor;
    };
    if constexpr (traits::is_sparse_matrix_v<H_t>) {
      for (Index outer = 0; outer < hessian.outerSize(); ++outer) {
        for (typename H_t::InnerIterator entry(hessian, outer); entry; ++entry)
          scale_entry(entry.valueRef(), entry.row(), entry.col());
      }
    } else {
      for (Index column = 0; column < hessian.cols(); ++column)
        for (Index row = 0; row < hessian.rows(); ++row)
          scale_entry(hessian(row, column), row, column);
    }
  }

  const Options options_;
  Grad_t scaling_;
  Scalar lambda_ = 1e-4f;              ///< Initial damping factor  (\lambda)
  Scalar prev_lambda_ = 0.0f;          ///< Previous damping factor  (0 at start)
  Scalar bad_factor_ = 2.0f;           ///< Current damping scaling factor for bad steps
  bool rebuild_linear_system_ = true;  ///< Whether the linear system (H and gradient) have to be
                                       ///< rebuilt or a simple evaluation can do it.
};

}  // namespace tinyopt::solvers
