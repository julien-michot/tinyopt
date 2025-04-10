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

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <limits>
#include <type_traits>
#include "tinyopt/traits.h"

#include <tinyopt/log.h>
#include <tinyopt/math.h>
#include <tinyopt/output.h>

#include <tinyopt/solvers/options.h>

namespace tinyopt::solvers {

template <typename _Scalar, int _Dims>
class SolverBase {
 public:
  using Scalar = _Scalar;
  static constexpr int Dims = _Dims;

  SolverBase(const solvers::Options1 &options = {}) : options_{options} {}

  /// Accumulate residuals and get the current error
  template <typename X_t, typename AccFunc, typename Hessian_t = std::nullptr_t>
  inline Scalar Evalulate(const X_t &x, const AccFunc &acc) {
    std::nullptr_t nul;
    if constexpr (traits::is_nullptr_v<Hessian_t>)
      this->Accumulate(x, acc, nul);
    else if constexpr (std::is_invocable_v<AccFunc, const X_t &, std::nullptr_t &,
                                           std::nullptr_t &>)
      this->Accumulate(x, acc, nul, nul);
    else {
      if (options_.log.enable)
        TINYOPT_LOG("⚠️ Your cost function doesn't support a nullptr Hessian, using a dummy {}",
                    typeid(Hessian_t).name());
      Hessian_t H;  // Dummy Hessian
      this->Accumulate(x, acc, nul, H);
    }
    return err_;
  }

 protected:
  /// Accumulate residuals and update the gradient, returns true on success
  template <typename X_t, typename AccFunc, typename Gradient_t,
            typename Hessian_t = std::nullptr_t>
  inline bool Accumulate(const X_t &x, const AccFunc &acc, Gradient_t &grad,
                         Hessian_t &H = SuperNul()) {
    if constexpr (std::is_invocable_v<AccFunc, const X_t &, Gradient_t &>)
      return ParseErrors(acc(x, grad));
    else
      return ParseErrors(acc(x, grad, H));
  }

  /// Extract errors and number of residuals from the output of accumulation function
  template <typename ErrorsType>
  inline bool ParseErrors(const ErrorsType &output) {
    if constexpr (traits::is_pair_v<ErrorsType>) {
      err_ = std::get<0>(output);
      nerr_ = std::get<1>(output);
    } else if constexpr (std::is_scalar_v<ErrorsType>) {
      err_ = output;
      nerr_ = 1;
    } else if constexpr (traits::is_matrix_or_array_v<ErrorsType>) {
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

  /// @brief Clamp the gradient 'g' to within [-minmax, minmax], if minmax is not 0.
  /// Returns true if 'g' was clamped.
  template <typename Grad_t>
  bool Clamp(Grad_t &g, Scalar minmax) const {
    if (minmax == 0) return false;
    if constexpr (std::is_scalar_v<Grad_t>) {
      g = std::clamp(g, -minmax, minmax);
    } else {
      g = g.cwiseMax(-minmax).cwiseMin(minmax);
    }
    return true;
  }

 public:
  /// Solve the linear system dx = -H^-1 * grad, returns nullopt on failure
  virtual std::optional<Vector<Scalar, Dims>> Solve() const = 0;

  virtual void Succeeded(Scalar = 0) {}
  virtual void Failed(Scalar = 0) {}

  virtual std::string stateAsString() const { return ""; }

  Scalar Error() const { return err_; }
  Scalar NumResiduals() const { return nerr_; }

 protected:
  const solvers::Options1 options_;
  Scalar err_ = std::numeric_limits<Scalar>::max();
  int nerr_ = 0;
};

}  // namespace tinyopt::solvers
