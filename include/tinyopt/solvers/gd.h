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

namespace tinyopt::gd {

struct SolverOptions : solvers::Options1 {
  SolverOptions(const solvers::Options1 &options = {}) : solvers::Options1{options} {}
  float lr = 1;  ///< Learning rate. The step dx will be -lr * gradient.
};

}  // namespace tinyopt::gd

namespace tinyopt::solvers {

template <typename Gradient_t = VecX>
class SolverGD
    : public SolverBase<typename Gradient_t::Scalar, traits::params_trait<Gradient_t>::Dims> {
 public:
  static constexpr bool IsNLLS = false;  // by default it is not
  static constexpr bool FirstOrder = true;
  using Base = SolverBase<typename Gradient_t::Scalar, traits::params_trait<Gradient_t>::Dims>;
  using Scalar = typename Gradient_t::Scalar;
  static constexpr Index Dims = traits::params_trait<Gradient_t>::Dims;
  // Gradient Type
  using Grad_t = Gradient_t;
  // Hessian Type (here none, so nullptr_t)
  using H_t = std::nullptr_t;
  // Options
  using Options = gd::SolverOptions;

  explicit SolverGD(const Options &options = {}) : Base(options), options_{options} {}

  /// Initialize solver with specific gradient and hessian
  void InitWith(const Grad_t &g) { grad_ = g; }

  /// Reset the solver state and clear gradient & hessian
  void reset() { clear(); }

  /// Resize H and grad if needed, return true if they were resized
  template <int D = Dims, std::enable_if_t<D == Dynamic, int> = 0>
  bool resize(int dims) {
    if (dims == Dynamic) {
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

  /// Ugly function to make sure the returned type is always a pair<scalar, scalar>...
  template <typename AccFunc>
  inline auto GetAccFunc(const AccFunc &acc) const {
    return [&](const auto &x, auto &grad) -> Cost {
      using X_t = decltype(x);
      if constexpr (std::is_invocable_v<AccFunc, const X_t &, std::nullptr_t &>)
        return acc(x, grad);
      else {
        std::nullptr_t nul;
        return acc(x, grad, nul);
      }
    };
  }

  /// Accumulate residuals and return the final error
  template <typename X_t, typename AccFunc>
  inline Scalar Evaluate(const X_t &x, const AccFunc &acc, bool save) {
    std::nullptr_t nul;
    const auto acc2 = GetAccFunc(acc);
    const auto &cost = acc2(x, nul);
    this->NormalizeCost(cost);
    if (save) this->cost_ = cost;
    return cost.cost;
  }

  /// Accumulate residuals and update the gradient, returns true on success
  template <typename X_t, typename AccFunc>
  inline bool Accumulate(const X_t &x, const AccFunc &acc) {
    const auto acc2 = GetAccFunc(acc);
    this->cost_ = acc2(x, grad_);
    this->NormalizeCost(this->cost_);
    return this->cost_.isValid();
  }

  /// Build the gradient and hessian by accumulating residuals and their jacobians
  /// Returns true on success
  template <typename X_t, typename AccFunc>
  inline bool Build(const X_t &x, const AccFunc &acc, bool resize_and_clear = true) {
    // Resize the system if needed and clear gradient
    if (resize_and_clear) {
      ResizeIfNeeded(x);
      clear();
    }
    // Update gradient by accumulating changes
    const bool ok = Accumulate(x, acc);
    // Eventually clip the gradients
    this->Clamp(grad_, options_.grad_clipping);
    return ok;
  }

  /// Solve the linear system dx = -lr * grad, returns nullopt on failure
  inline std::optional<Vector<Scalar, Dims>> Solve() const override {
    if (!this->cost().isValid()) return std::nullopt;
    return -options_.lr * grad_;
  }

  Index dims() const override { return grad_.size(); }

  const Grad_t &Gradient() const { return grad_; }
  Grad_t &Gradient() { return grad_; }
  Scalar GradientNorm() const { return grad_.norm(); }
  Scalar GradientSquaredNorm() const { return grad_.squaredNorm(); }

 protected:
  const Options options_;
  Grad_t grad_;
};

}  // namespace tinyopt::solvers
