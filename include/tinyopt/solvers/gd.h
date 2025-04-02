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

#include <tinyopt/log.h>
#include <tinyopt/math.h>
#include <tinyopt/options.h>
#include <tinyopt/output.h>
#include <tinyopt/time.h>
#include <tinyopt/traits.h>

#include <tinyopt/optimize_jet.h>
#include <tinyopt/solvers/options.h>

namespace tinyopt::gd {

struct SolverOptions : solvers::Solver1Options {
  SolverOptions(const solvers::Solver1Options &options = {}) : solvers::Solver1Options{options} {}
  float lr = 1e-3;  ///< Learning rate. The step dx will be -lr * gradient.
};

}  // namespace tinyopt::gd

namespace tinyopt::solvers {

template <typename Gradient_t = VecX>
class SolverGD {
 public:
  static constexpr bool FirstOrder = true;
  using Scalar = typename Gradient_t::Scalar;
  static constexpr int Dims = traits::params_trait<Gradient_t>::Dims;
  // Gradient Type
  using Grad_t = Gradient_t;
  // Options
  using Options = gd::SolverOptions;

  explicit SolverGD(const Options &options = {}) : options_{options} {}

  /// Initialize solver with specific gradient and hessian
  void InitWith(const Grad_t &g) { grad_ = g; }

  /// Reset the solver state and clear gradient & hessian
  void reset() { clear(); }

  /// Resize H and grad if needed, return true if they were resized
  template <int D = Dims, std::enable_if_t<D == Dynamic, int> = 0>
  bool resize(int dims) {
    if (dims == Dynamic) {
      TINYOPT_LOG("Error: Dimensions cannot be Dynamic here");
      throw std::invalid_argument("Dimensions cannot be Dynamic here");
    }
    if (grad_.rows() != dims) {
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
    return false;
  }

  /// Set gradient and hessian to 0s
  void clear() { grad_.setZero(); }

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
    const auto &output = acc(x, grad_);

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
    // Solver linear system
    if (nerr_ > 0) {
      dx = -options_.lr * grad_;
      return true;
    }
    if (!success) dx.setZero();
    return false;
  }

  void Succeeded(Scalar = 0) {}

  void Failed(Scalar = 0) {}

  constexpr std::string LogString() const { return ""; }

  const Grad_t &Gradient() const { return grad_; }
  Grad_t &Gradient() { return grad_; }
  Scalar GradientNorm() const { return grad_.norm(); }
  Scalar GradientSquaredNorm() const { return grad_.squaredNorm(); }

  Scalar Error() const { return err_; }
  Scalar NumResiduals() const { return nerr_; }

 protected:
  const Options options_;
  Grad_t grad_;
  Scalar err_ = std::numeric_limits<Scalar>::max();
  int nerr_ = 0;
};

}  // namespace tinyopt::solvers
