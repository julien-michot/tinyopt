// Copyright (C) 2025 Julien Michot. All Rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <tinyopt/cost.h>
#include <tinyopt/log.h>
#include <tinyopt/solvers/gn.h>
#include <tinyopt/solvers/options.h>

namespace tinyopt::nlls::lm {

/***
 *  @brief Levenberg-Marquardt Solver Optimization options
 *
 ***/
struct SolverOptions : tinyopt::nlls::gn::SolverOptions {
  SolverOptions(const tinyopt::nlls::gn::SolverOptions options = {})
      : tinyopt::nlls::gn::SolverOptions{options} {}

  /**
   * @name Damping options
   * @{
   */
  float damping_init = 1e-4f;  ///< Initial damping factor. If 0, the damping is disable (it will
                               ///< behave like Gauss-Newton)
  ///< Min and max damping values (only used when damping_init != 0)
  std::array<float, 2> damping_range{{1e-9f, 1e9f}};

  float good_factor = 1.0f / 3.0f;  ///< Scale to apply to the damping for good steps
  float bad_factor = 2.0f;          ///< Scale to apply to the damping for bad steps
  /** @} */
};
}  // namespace tinyopt::nlls::lm

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
  using Options = nlls::lm::SolverOptions;

  explicit SolverLM(const Options &options = {}) : Base(options), options_{options} {
    // Sparse matrix must use LDLT
    if constexpr (traits::is_sparse_matrix_v<H_t>) {
      if (!options.use_ldlt) TINYOPT_LOG("Warning: LDLT must be used with Sparse Matrices");
    }
    reset();
  }

  /// Reset the solver state and clear gradient & hessian
  void reset() override {
    this->clear();
    lambda_ = options_.damping_init;
    prev_lambda_ = 0;
    bad_factor_ = options_.bad_factor;
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
      this->Clamp(this->grad_, options_.grad_clipping);

      // Verify Hessian's diagonal
      if (options_.check_min_H_diag > 0 &&
          (this->H_.diagonal().cwiseAbs().array() < options_.check_min_H_diag).any()) {
        if (options_.log.enable) TINYOPT_LOG("❌ Hessian has very low diagonal coefficients");
        return false;
      }

      // Fill the lower part if H if needed
      if constexpr (!traits::is_sparse_matrix_v<H_t>) {
        if (!options_.H_is_full && !options_.use_ldlt) {
          this->H_.template triangularView<Lower>() =
              this->H_.template triangularView<Upper>().transpose();
        }
      }

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

    return true;
  }

  /// Damping stategy for a good step: increase the damping factor \lambda
  void GoodStep(Scalar quality) override {
    Scalar s = options_.good_factor;  // Scale to apply on damping lambda

    // Use an approximative scaling based on the step quality TODO: improve this
    if (quality != Scalar(0.0)) {
      s = std::max<Scalar>(s, 1.0f - std::pow(2.0f * quality - 1.0f, 3.0f));
    }

    // Check whether the previous 'bad' step was actually good and revert the last scaling
    if (bad_factor_ != options_.bad_factor) s /= bad_factor_;

    prev_lambda_ = lambda_;
    lambda_ = std::clamp<Scalar>(lambda_ * s, options_.damping_range[0], options_.damping_range[1]);
    bad_factor_ = options_.bad_factor;
  }

  /// Damping stategy for a bad step: decrease the damping factor \lambda
  void BadStep(Scalar /*quality*/ = 0.0f) override {
    Scalar s = bad_factor_;  // Scale to apply on damping lambda
    prev_lambda_ = lambda_;
    lambda_ = std::clamp<Scalar>(lambda_ * s, options_.damping_range[0], options_.damping_range[1]);
    bad_factor_ *= options_.bad_factor;
  }

  /// Damping stategy for a failure to solve the linear system, decrease the damping factor \lambda
  void FailedStep() override { BadStep(); }

  std::string stateAsString() const override {
    std::ostringstream oss;
    oss << TINYOPT_FORMAT_NS::format("○:{:.2e} ", 1.0 / lambda_);
    return oss.str();
  }

  /// Latest Hessian approximation (JtJ), un-damped
  H_t Hessian() const {
    if (prev_lambda_ > 0.0) {
      H_t H = this->H_;  // copy
      const Scalar s = 1.0f + prev_lambda_;
      for (int i = 0; i < this->H_.cols(); ++i) {
        if constexpr (traits::is_matrix_or_array_v<H_t>)
          H(i, i) /= s;
        else
          H.coeffRef(i, i) = this->H_.coeff(i, i) / s;
      }
      return H;
    } else {
      return this->H_;
    }
  }

  /// Latest Covariance estimate
  std::optional<H_t> Covariance() const override { return InvCov(Hessian()); }

  /// Return the square root of the maximum (co)variance of the H.inv()
  /// H being the damped Hessian H_ if use_damped == true (faster) or un-damped Hessian() (accurate)
  Scalar MaxStdDev(bool use_damped = true) const {
    const auto &H = use_damped ? this->H_ : Hessian();
    const auto I = InvCov(H);
    if (!I) return 0;
    using std::sqrt;
    if constexpr (traits::is_sparse_matrix_v<H_t>)
      return sqrt(I.value().coeffs().maxCoeff());
    else
      return sqrt(I.value().maxCoeff());
  }

 protected:
  const Options options_;
  Scalar lambda_ = 1e-4f;              ///< Initial damping factor  (\lambda)
  Scalar prev_lambda_ = 0.0f;          ///< Previous damping factor  (0 at start)
  Scalar bad_factor_ = 2.0f;           ///< Current damping scaling factor for bad steps
  bool rebuild_linear_system_ = true;  ///< Whether the linear system (H and gradient) have to be
                                       ///< rebuilt or a simple evaluation can do it.
};

}  // namespace tinyopt::solvers
