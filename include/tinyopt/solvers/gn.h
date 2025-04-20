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
#include <tinyopt/solvers/base.h>
#include <tinyopt/solvers/options.h>
#include <tinyopt/traits.h>

namespace tinyopt::nlls::gn {

/***
 *  @brief Gauss-Newton Solver Optimization options
 *
 ***/
using SolverOptions = solvers::Options2;

}  // namespace tinyopt::nlls::gn

namespace tinyopt::solvers {

template <typename Hessian_t = MatX>
class SolverGN
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
  using Options = nlls::gn::SolverOptions;

  explicit SolverGN(const Options &options = {}) : Base(options), options_{options} {
    // Sparse matrix must use LDLT
    if constexpr (traits::is_sparse_matrix_v<H_t>) {
      if (!options.use_ldlt) TINYOPT_LOG("Warning: LDLT must be used with Sparse Matrices");
    }
  }

  /// Initialize solver with specific gradient and hessian
  void InitWith(const Grad_t &g, const H_t &h) {
    grad_ = g;
    H_ = h;
  }

  /// Reset the solver state and clear gradient & hessian
  virtual void reset() { clear(); }

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
    H_.setZero();
    grad_.setZero();
  }

  /// Check whether we need to resize the system (gradient), return true if it did
  template <typename X_t>
  bool ResizeIfNeeded(const X_t &x) {
    if constexpr (Dims == Dynamic) {
      const int dims = traits::params_trait<X_t>::dims(x);
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
    // Resize the system if needed and clear gradient
    if (resize_and_clear) {
      ResizeIfNeeded(x);
      clear();
    }

    // Accumulate residuals and update both gardient and Hessian approx (Jt*J)
    const bool success = Accumulate(x, res_func);
    if (!success) return false;

    // Eventually clip the gradient
    this->Clamp(grad_, options_.grad_clipping);

    // Verify Hessian's diagonal
    if (options_.check_min_H_diag > 0 &&
        (H_.diagonal().cwiseAbs().array() < options_.check_min_H_diag).any()) {
      if (options_.log.enable) TINYOPT_LOG("❌ Hessian has very low diagonal coefficients");
      return false;
    }

    // Fill the lower part if H if needed
    {
      if (!options_.H_is_full && !options_.use_ldlt) {
        H_.template triangularView<Lower>() = H_.template triangularView<Upper>().transpose();
      }
    }
    return true;
  }

  /// Solve the linear system dx = -H^-1 * grad, returns nullopt on failure
  inline std::optional<Vector<Scalar, Dims>> Solve() const override {
    if (!this->cost().isValid()) return std::nullopt;

    // Solver linear system
    if (options_.use_ldlt || traits::is_sparse_matrix_v<H_t>) {
      const auto dx_ = tinyopt::SolveLDLT(H_, -grad_);
      if (dx_) return dx_;                                    // Hopefully not a copy...
    } else if constexpr (!traits::is_sparse_matrix_v<H_t>) {  // Use default inverse
      if constexpr (Dims == 1) {
        if (H_(0, 0) > FloatEpsilon<Scalar>()) return -H_.inverse() * grad_;
        return Vector<Scalar, Dims>::Zero(grad_.size());
      } else
        return -H_.inverse() * grad_;
    }
    // Log on failure
    if (options_.log.enable && options_.log.print_failure) {
      TINYOPT_LOG("❌ Failed solve linear system");
      TINYOPT_LOG("grad = \n{}", grad_);
      TINYOPT_LOG("H = \n{}", H_);
    }
    return std::nullopt;
  }

  /// Latest Hessian approximation (JtJ), un-damped
  const H_t &Hessian() const { return H_; }

  /// Return the square root of the maximum (co)variance of the H.inv()
  /// H being the damped Hessian H_ if use_damped == true (faster) or un-damped Hessian() (accurate)
  Scalar MaxStdDev(bool use_damped = true) const {
    const auto &H = use_damped ? H_ : Hessian();
    if constexpr (traits::is_sparse_matrix_v<H_t>)
      return sqrt(InvCov(H).value().coeffs().maxCoeff());
    else
      return sqrt(InvCov(H).value().maxCoeff());
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
  H_t H_;
  Grad_t grad_;
};

}  // namespace tinyopt::solvers
