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

#include <tinyopt/solvers/base.h>
#include <tinyopt/solvers/options.h>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include "tinyopt/log.h"

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

  /** @} */
};
}  // namespace tinyopt::nlls::lm

namespace tinyopt::solvers {

template <typename Hessian_t = MatX>
class SolverLM
    : public SolverBase<typename Hessian_t::Scalar, SQRT(traits::params_trait<Hessian_t>::Dims)> {
 public:
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
    lambda_ = static_cast<Scalar>(options.damping_init);
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
  void reset() {
    lambda_ = options_.damping_init;
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
    H_.setZero();
    grad_.setZero();
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

  /// Accumulate residuals and update the gradient, returns true on success
  template <typename ResidualsFunc>
  inline auto GetAccFunc(const ResidualsFunc &res_func) const {
    // Get final error
    return [&](const auto &x, auto &grad, auto &H) {
      const auto &output = res_func(x, grad, H);
      using ErrorType = std::decay_t<decltype(output)>;
      if constexpr (std::is_scalar_v<ErrorType>) {
        return std::make_pair(output, 1);
      } else if constexpr (traits::is_pair_v<ErrorType>) {
        return output;
      } else {  // must be a Vector/Matrix/Array
        static_assert(traits::is_matrix_or_array_v<ErrorType>, "Unknown returned type");
        return std::make_pair(output.norm(), output.size());  // return L2/Frobenius norm
      }
    };
  }

  /// Accumulate residuals and update the gradient, returns true on success
  template <typename X_t, typename ResidualsFunc>
  inline Scalar Evaluate(const X_t &x, const ResidualsFunc &res_func) const {
    std::nullptr_t nul;
    const auto acc = GetAccFunc(res_func);
    if constexpr (std::is_invocable_v<ResidualsFunc, const X_t &, std::nullptr_t &,
                                      std::nullptr_t &>)
      return acc(x, nul, nul).first;
    else {
      Hessian_t H;  // dummy;
      if (options_.log.enable)
        TINYOPT_LOG("⚠️ Your cost function doesn't support a nullptr Hessian, using a dummy {}",
                    typeid(Hessian_t).name());
      return acc(x, nul, H).first;
    }
  }

  /// Accumulate residuals and update the gradient, returns true on success
  template <typename X_t, typename ResidualsFunc>
  inline bool Accumulate(const X_t &x, const ResidualsFunc &res_func) {
    const auto acc = GetAccFunc(res_func);
    const auto &[e, ne] = acc(x, grad_, H_);
    this->err_ = e;
    this->nerr_ = static_cast<int>(ne);
    return ne > 0;
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

    // Damping
    if (lambda_ > 0.0) {
      for (int i = 0; i < H_.rows(); ++i) {
        if constexpr (traits::is_matrix_or_array_v<H_t>)
          H_(i, i) *= 1.0f + lambda_;
        else {
          H_.coeffRef(i, i) *= 1.0f + lambda_;
        }
      }
    }

    // Fill the lower part if H if needed
    if constexpr (!traits::is_sparse_matrix_v<H_t>) {
      if (!options_.H_is_full && !options_.use_ldlt) {
        H_.template triangularView<Lower>() = H_.template triangularView<Upper>().transpose();
      }
    }
    return true;
  }

  /// Solve the linear system dx = -H^-1 * grad, returns nullopt on failure
  std::optional<Vector<Scalar, Dims>> Solve() const override {
    if (this->nerr_ == 0) return std::nullopt;

    // Solver linear system
    if (options_.use_ldlt || traits::is_sparse_matrix_v<H_t>) {
      const auto dx_ = tinyopt::SolveLDLT(H_, grad_);
      if (dx_) {
        return -dx_.value();
      }
    } else if constexpr (!traits::is_sparse_matrix_v<H_t>) {  // Use default inverse
      return -H_.inverse() * grad_;
    }
    return std::nullopt;
  }

  void Succeeded(Scalar scale = 1.0f / 3.0f) override {
    if (lambda_ == 0.0f) return;
    lambda_ =
        std::clamp<Scalar>(lambda_ * scale, options_.damping_range[0], options_.damping_range[1]);
  }

  void Failed(Scalar scale = 2.0f) override {
    if (lambda_ == 0.0f) return;
    lambda_ =
        std::clamp<Scalar>(lambda_ * scale, options_.damping_range[0], options_.damping_range[1]);
  }

  std::string stateAsString() const override {
    std::ostringstream oss;
    oss << TINYOPT_FORMAT_NAMESPACE::format("λ:{:.2e} ", lambda_);
    return oss.str();
  }

  /// Latest Hessian approximation (JtJ), un-damped
  auto Hessian() const {
    if (lambda_ > 0.0) {
      H_t H = H_;  // copy
      for (int i = 0; i < H_.cols(); ++i) {
        if constexpr (traits::is_matrix_or_array_v<H_t>)
          H(i, i) = H_(i, i) / (1.0f + lambda_);
        else
          H.coeffRef(i, i) = H_.coeff(i, i) / (1.0f + lambda_);
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
  H_t H_;
  Grad_t grad_;
  Scalar lambda_ = 0;
};

}  // namespace tinyopt::solvers
