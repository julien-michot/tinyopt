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

  SolverBase() {}

 protected:
  /// Accumulate residuals and update the gradient, returns true on success
  template <typename X_t, typename AccFunc_t, typename Gradient_t>
  inline bool Accumulate1(const X_t &x, const AccFunc_t &acc, Gradient_t &grad) {
    // Update gradient by accumulating changes
    if constexpr (std::is_invocable_v<AccFunc_t, const X_t &, Gradient_t &>) {
      return ParseErrors(acc(x, grad));
    } else {
      std::nullptr_t nulle = nullptr;
      return ParseErrors(acc(x, grad, nulle));
    }
  }

  /// Accumulate residuals and update the gradient, returns true on success
  template <typename X_t, typename AccFunc_t, typename Gradient_t, typename Hessian_t>
  inline bool Accumulate2(const X_t &x, const AccFunc_t &acc, Gradient_t &grad, Hessian_t &H) {
    static_assert(std::is_invocable_v<AccFunc_t, const X_t &, Gradient_t &, Hessian_t &>);
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

 public:
  /// Solve the linear system dx = -H^-1 * grad, returns nullopt on failure
  virtual std::optional<Vector<Scalar, Dims>> Solve() const = 0;

  virtual void Succeeded(Scalar = 0) {}
  virtual void Failed(Scalar = 0) {}

  virtual std::string stateAsString() const { return ""; }

  Scalar Error() const { return err_; }
  Scalar NumResiduals() const { return nerr_; }

 protected:
  Scalar err_ = std::numeric_limits<Scalar>::max();
  int nerr_ = 0;
};

}  // namespace tinyopt::solvers
