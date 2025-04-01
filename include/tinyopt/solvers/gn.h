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

#include <cassert>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include <tinyopt/log.h>
#include <tinyopt/math.h>
#include <tinyopt/opt_jet.h>
#include <tinyopt/options.h>
#include <tinyopt/output.h>
#include <tinyopt/time.h>
#include <tinyopt/traits.h>

#include <tinyopt/solvers/options.h>

namespace tinyopt::solvers {

namespace gn {
using SolverOptions = solvers::Solver2Options;
}  // namespace gn

template <typename HessianMatrixType = MatX>
class SolverGN {
 public:
  static constexpr bool FirstOrder = false;  // this is a pseudo second order algorithm
  using Scalar = typename HessianMatrixType::Scalar;
  static constexpr int Dims = traits::params_trait<HessianMatrixType>::Dims == Dynamic
                                  ? Dynamic
                                  : SQRT(traits::params_trait<HessianMatrixType>::Dims);
  // Hessian Type
  using H_t = HessianMatrixType;
  // Gradient Type
  using Grad_t = Vector<Scalar, Dims>;
  // Options
  using Options = gn::SolverOptions;

  explicit SolverGN(const Options &options = {}) : options_{options} {}

  /// Initialize solver with specific gradient and hessian
  void InitWith(const Grad_t &g, const H_t &h) {
    grad_ = g;
    H_ = h;
  }

  /// Reset the solver state and clear gradient & hessian
  void reset() { clear(); }

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
      clear();
      return true;
    } else {
      return false;
    }
  }

  /// Resize H and grad if needed, return true if they were resized
  template <int D = Dims, std::enable_if_t<D != Dynamic, int> = 0>
  bool resize(int dims = Dims) {
    if (dims == Dynamic) {
      TINYOPT_LOG("Error: Static and Dynamic Dimensions must match");
      throw std::invalid_argument("Error: Static and Dynamic Dimensions must match");
    }
    if constexpr (traits::is_sparse_matrix_v<H_t>) {
      H_.resize(dims, dims);
      grad_.resize(dims);
      clear();
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

  /// Build Gradient and Hessian and solve the linear system H * x = g
  /// Returns true on success
  template <typename X_t, typename AccFunc>  // TODO std::function
  inline bool Solve(const X_t &x, const AccFunc &acc, Vector<Scalar, Dims> &dx) {
    using std::sqrt;
    int dims = Dims;  // Dynamic size
    if constexpr (Dims == Dynamic) dims = traits::params_trait<X_t>::dims(x);
    if (dims == Dynamic || dims == 0) {
      if (options_.log.enable) TINYOPT_LOG("❌ Nothing to optimize");
      return false;
    }

    // Clear the system TODO do not clear if InitWith was called or do that outside
    clear();

    // Update Hessian approx and gradient by accumulating changes
    const auto &output = acc(x, grad_, H_);

    // Recover final error TODO clean this
    using ResOutputType = std::remove_const_t<std::remove_reference_t<decltype(output)>>;
    if constexpr (traits::is_pair_v<ResOutputType>) {
      using ResOutputType1 =
          std::remove_const_t<std::remove_reference_t<decltype(std::get<0>(output))>>;
      if constexpr (traits::is_matrix_or_array_v<ResOutputType1>) {
        err_ = std::get<0>(output).squaredNorm();
        if (std::get<0>(output).size() == 0) nerr_ = 0;
      } else {
        err_ = std::get<0>(output);
      }
      nerr_ = std::get<1>(output);
    } else if constexpr (std::is_scalar_v<ResOutputType>) {
      err_ = output;
      nerr_ = 1;
    } else if constexpr (traits::is_matrix_or_array_v<ResOutputType>) {
      err_ = output.squaredNorm();
      if (output.size() == 0) nerr_ = 0;
    } else {
      // You're not returning a supported type (must be float, double or Matrix)
      static_assert(traits::is_matrix_or_array_v<ResOutputType> || std::is_scalar_v<ResOutputType>);
    }

    bool success = nerr_ > 0;
    if (success) {  // ok we got residuals

      // Solver linear system
      if (options_.ldlt || traits::is_sparse_matrix_v<H_t>) {
        const auto dx_ = tinyopt::Solve(H_, grad_);
        if (dx_) {
          dx = -dx_.value();
          success = true;
        }
      } else if constexpr (!traits::is_sparse_matrix_v<H_t>) {  // Use default inverse
        // Fill the lower part of H then inverse it
        if (!options_.H_is_full)
          H_.template triangularView<Lower>() = H_.template triangularView<Upper>().transpose();
        dx = -H_.inverse() * grad_;
        success = true;
      }
    }

    if (!success) dx.setZero();
    return success;
  }

  void Succeeded(Scalar = 0) {}

  void Failed(Scalar = 0) {}

  constexpr std::string LogString() const { return ""; }

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

  Scalar Error() const { return err_; }
  Scalar NumResiduals() const { return nerr_; }

 protected:
  const Options options_;
  H_t H_;
  Grad_t grad_;
  Scalar err_ = std::numeric_limits<Scalar>::max();
  int nerr_ = 0;
};

}  // namespace tinyopt::solvers
