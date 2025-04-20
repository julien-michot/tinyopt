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

#include <cstddef>
#include <stdexcept>

#include <tinyopt/cost.h>
#include <tinyopt/log.h>
#include <tinyopt/solvers/base.h>
#include <tinyopt/solvers/options.h>

namespace tinyopt::nlls::lm {

/***
 *  @brief Levenberg-Marquardt Solver Optimization options
 *
 ***/
struct SolverOptions : solvers::Options2 {
  SolverOptions(const solvers::Options2 options = {}) : solvers::Options2{options} {}

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
class SolverLM
    : public SolverBase<typename Hessian_t::Scalar, SQRT(traits::params_trait<Hessian_t>::Dims)> {
 public:
  static constexpr bool IsNLLS = true;
  static constexpr bool FirstOrder = false;  // this is a pseudo second order algorithm
  using Base = SolverBase<typename Hessian_t::Scalar, SQRT(traits::params_trait<Hessian_t>::Dims)>;
  using Scalar = typename Hessian_t::Scalar;
  static constexpr Index Dims = SQRT(traits::params_trait<Hessian_t>::Dims);

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

  /// Initialize solver with specific gradient and hessian
  void InitWith(const Grad_t &g, const H_t &h) {
    grad_ = g;
    H_ = h;
  }

  /// Reset the solver state and clear gradient & hessian
  void reset() {
    lambda_ = static_cast<Scalar>(options_.damping_init);
    prev_lambda_ = static_cast<Scalar>(0.0f);
    bad_factor_ = static_cast<Scalar>(options_.bad_factor);
    steps_count_ = 0;
    clear();
  }

  /// Resize H and grad if needed, return true if they were resized
  template <int D = Dims, std::enable_if_t<D == Dynamic, int> = 0>
  bool resize(int dims) {
    if (dims == Dynamic) {
      TINYOPT_LOG("Error: Dimensions cannot be Dynamic here");
      throw std::invalid_argument("Dimensions cannot be Dynamic here");
    }
    if (grad_.rows() != dims || H_.rows() != dims) {
      H_.resize(dims, dims);
      grad_.resize(dims);
      return true;
    } else {
      return false;
    }
  }

  /// Resize H and grad if needed, return true if they were resized
  template <int D = Dims, std::enable_if_t<D != Dynamic, int> = 0>
  bool resize(int dims = Dims) {
    if (dims != Dims) {
      TINYOPT_LOG("Error: Static and Dynamic Dimensions must match");
      throw std::invalid_argument("Error: Static and Dynamic Dimensions must match");
    }
    if constexpr (traits::is_sparse_matrix_v<H_t>) {
      H_.resize(dims, dims);
      grad_.resize(dims);
      return true;
    }
    return false;
  }

  /// Set gradient and hessian to 0s
  void clear() {
    // Fill H & grad fill 0s (not needed when using auto-jet)
    if (grad_.size()) grad_.setZero();
    if (H_.size()) H_.setZero();
  }

  /// Check whether we need to resize the system (gradient), return true if it did
  template <typename X_t>
  bool ResizeIfNeeded(const X_t &x) {
    if constexpr (Dims == Dynamic) {
      const auto dims = traits::params_trait<X_t>::dims(x);
      if (grad_.rows() != dims) {
        if (options_.log.enable) TINYOPT_LOG("Need to resize the system");
        return resize(dims);
      }
    }
    return false;
  }

  /// Accumulate residuals and return the final error
  template <typename X_t, typename AccFunc>
  inline Scalar Evaluate(const X_t &x, const AccFunc &acc, bool save) {
    std::nullptr_t nul;
    Hessian_t H;  // dummy;
    Cost cost = acc(x, nul, H);
    this->NormalizeCost(cost);
    if (save) this->cost_ = cost;
    return cost.cost;
  }

  /// Accumulate residuals and update the gradient, returns true on success
  template <typename X_t, typename AccFunc>
  inline bool Accumulate(const X_t &x, const AccFunc &acc) {
    this->cost_ = acc(x, grad_, H_);
    this->NormalizeCost(this->cost_);
    return this->cost_.isValid();
  }

  /// Build the gradient and hessian by accumulating residuals and their jacobians
  /// Returns true on success
  template <typename X_t, typename ResidualsFunc>
  inline bool Build(const X_t &x, const ResidualsFunc &res_func, bool resize_and_clear = true) {
    if (rebuild_linear_system_) {
      // Resize the system if needed and clear gradient
      if (resize_and_clear) {
        ResizeIfNeeded(x);
        clear();
      }

      // Accumulate residuals and update both gardient and Hessian approx (Jt*J)
      const bool success = Accumulate(x, res_func);

      // Early skip on failure (no residuals)
      if (!success) {
        if (options_.log.enable) TINYOPT_LOG("❌ Failed to accumulate residuals: {}", this->cost().toString());
        return false;
      }

      // Eventually clip the gradient
      this->Clamp(grad_, options_.grad_clipping);

      // Verify Hessian's diagonal
      if (options_.check_min_H_diag > 0 &&
          (H_.diagonal().cwiseAbs().array() < options_.check_min_H_diag).any()) {
        if (options_.log.enable) TINYOPT_LOG("❌ Hessian has very low diagonal coefficients");
        return false;
      }

      // Fill the lower part if H if needed
      if constexpr (!traits::is_sparse_matrix_v<H_t>) {
        if (!options_.H_is_full && !options_.use_ldlt) {
          H_.template triangularView<Lower>() = H_.template triangularView<Upper>().transpose();
        }
      }

    } else {  // Keeping H and gradient, only evaluate the cost again

      Evaluate(x, res_func, true);
      const bool success = this->cost().isValid();
      // Early skip on failure (no residuals)
      if (!success) {
        if (options_.log.enable) TINYOPT_LOG("❌ Failed to accumulate residuals");
        return false;
      }
    }

    // Damping
    if (lambda_ > 0.0) {
      const double s =
          rebuild_linear_system_ ? 1.0 + lambda_ : (1.0 + lambda_) / (1.0 + prev_lambda_);
      for (int i = 0; i < H_.rows(); ++i) {
        if constexpr (traits::is_matrix_or_array_v<H_t>)
          H_(i, i) *= s;
        else
          H_.coeffRef(i, i) *= s;
      }
    }

    return true;
  }

  /// Solve the linear system dx = -H^-1 * grad, returns nullopt on failure
  std::optional<Vector<Scalar, Dims>> Solve() const override {
    if (!this->cost().isValid()) return std::nullopt;

    // Solve the linear system
    if (options_.use_ldlt || traits::is_sparse_matrix_v<H_t>) {
      const auto dx_ = tinyopt::SolveLDLT(H_, -grad_);
      if (dx_) return dx_;                                    // Hopefully not a copy...
    } else if constexpr (!traits::is_sparse_matrix_v<H_t>) {  // Use default inverse
      if constexpr (Dims == 1) {
        if (H_(0, 0) > FloatEpsilon<Scalar>()) return -H_.inverse() * grad_;
        return Vector<Scalar, Dims>::Zero(grad_.size());
      } else {
        return -H_.inverse() * grad_;
      }
    }
    // Log on failure
    if (options_.log.enable && options_.log.print_failure) {
      TINYOPT_LOG("❌ Failed solve linear system, {}", stateAsString());
      TINYOPT_LOG("grad = \n{}", grad_);
      TINYOPT_LOG("H = \n{}", H_);
    }
    return std::nullopt;
  }

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
    if (steps_count_ < 3) steps_count_++;
  }

  void BadStep(Scalar /*quality*/ = 0.0f) override {
    Scalar s = bad_factor_;  // Scale to apply on damping lambda

    // Check whether the very first step was actually wrong and revert the scale applied to lambda
    if (steps_count_ == 1) s /= options_.good_factor;

    prev_lambda_ = lambda_;
    lambda_ = std::clamp<Scalar>(lambda_ * s, options_.damping_range[0], options_.damping_range[1]);
    bad_factor_ *= options_.bad_factor;
    if (steps_count_ < 3) steps_count_++;
  }

  void FailedStep() override { BadStep(); }

  void Rebuild(bool b) override { rebuild_linear_system_ = b; }

  std::string stateAsString() const override {
    std::ostringstream oss;
    oss << TINYOPT_FORMAT_NS::format("○:{:.2e} ", 1.0 / lambda_);
    return oss.str();
  }

  /// Latest Hessian approximation (JtJ), un-damped
  auto Hessian() const {
    if (prev_lambda_ > 0.0) {
      H_t H = H_;  // copy
      const Scalar s = 1.0f + prev_lambda_;
      for (int i = 0; i < H_.cols(); ++i) {
        if constexpr (traits::is_matrix_or_array_v<H_t>)
          H(i, i) /= s;
        else
          H.coeffRef(i, i) = H_.coeff(i, i) / s;
      }
      return H;
    } else {
      return H_;
    }
  }

  /// Return the square root of the maximum (co)variance of the H.inv()
  /// H being the damped Hessian H_ if use_damped == true (faster) or un-damped Hessian() (accurate)
  Scalar MaxStdDev(bool use_damped = true) const {
    const auto &H = use_damped ? H_ : Hessian();
    const auto I = InvCov(H);
    if (!I) return 0;
    using std::sqrt;
    if constexpr (traits::is_sparse_matrix_v<H_t>)
      return sqrt(I.value().coeffs().maxCoeff());
    else
      return sqrt(I.value().maxCoeff());
  }

  /// Latest, eventually damped Hessian approximation (JtJ)
  const H_t &H() const { return H_; }
  H_t &H() { return H_; }

  const Grad_t &Gradient() const { return grad_; }
  Grad_t &Gradient() { return grad_; }
  Scalar GradientNorm() const { return grad_.norm(); }
  Scalar GradientSquaredNorm() const { return grad_.squaredNorm(); }

 protected:
  const Options options_;
  H_t H_;        ///< Hessian Approximate (Jt*J)
  Grad_t grad_;  ///< Gradient (Jt*residuals)
  Scalar lambda_ = 1e-4f;
  Scalar prev_lambda_ = 0;  // 0 at start
  Scalar bad_factor_ = 2.0f;
  int steps_count_ = 0;  // Count step until 2
  bool rebuild_linear_system_ = true;
};

}  // namespace tinyopt::solvers
