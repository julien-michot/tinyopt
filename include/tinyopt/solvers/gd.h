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
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>

#include <tinyopt/log.h>
#include <tinyopt/math.h>
#include <tinyopt/output.h>
#include <tinyopt/traits.h>

#include <tinyopt/optimize_jet.h>
#include <tinyopt/solvers/options.h>

namespace tinyopt::gd {

struct SolverOptions : solvers::Options1 {
  SolverOptions(const solvers::Options1 &options = {}) : solvers::Options1{options} {}
  float lr = 1;  ///< Learning rate. The step dx will be -lr * gradient.
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
  // Hessian Type (here none, so nullptr_t)
  using H_t = std::nullptr_t;
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
    if constexpr (traits::is_sparse_matrix_v<Grad_t>) {
      grad_.resize(dims);
      return true;
    }
    return false;
  }

  /// Set gradient and hessian to 0s
  void clear() { grad_.setZero(); }

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

  /// Build the gradient and hessian by accumulating residuals and their jacobians
  /// Returns true on success
  template <typename X_t, typename AccFunc>  // TODO std::function
  inline bool Build(const X_t &x, const AccFunc &acc, bool resize_and_clear = true) {
    // Resize the system if needed and clear gradient
    if (resize_and_clear) {
      ResizeIfNeeded(x);
      clear();
    }

    // Update Hessian approx and gradient by accumulating changes
    const auto &output = acc(x, grad_);

    // Recover final error TODO clean this
    using ResOutputType = std::remove_const_t<std::remove_reference_t<decltype(output)>>;
    if constexpr (traits::is_pair_v<ResOutputType>) {
      err_ = std::get<0>(output);
      nerr_ = std::get<1>(output);
    } else if constexpr (std::is_scalar_v<ResOutputType>) {
      err_ = output;
      nerr_ = 1;
    } else if constexpr (traits::is_matrix_or_array_v<ResOutputType>) {
      err_ = output.norm();  // L2 or Frobenius
      nerr_ = output.size();
    } else {
      // You're not returning a supported type (must be float, double or Matrix)
      // TODO static_assert(false); // fails on MacOS...
      TINYOPT_LOG("❌ The loss returns a unknown type.");
      return false;
    }
    return true;
  }

  /// Solve the linear system dx = -lr * grad, returns nullopt on failure
  inline std::optional<Vector<Scalar, Dims>> Solve() const {
    if (nerr_ == 0) return std::nullopt;
    return -options_.lr * grad_;
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
