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

#include <tinyopt/diff/num_diff.h>
#include <tinyopt/math.h>

#include <tinyopt/log.h>
#include <tinyopt/traits.h>
#include <type_traits>

namespace tinyopt::diff {

/// @brief Compares the gradient computed by the user-provided 'acc' function with a numerically
///        estimated gradient.  This function helps verify the correctness of the analytical
///        gradient calculation in your cost/objective function.
///
/// @tparam X_t         Type of the input variable `x`.  This could be an Eigen matrix/vector type
///                     or a standard container like std::vector.  It must support basic arithmetic
///                     operations and element access.
/// @tparam AccFunc     Type of the 'acc' function (which calculates the cost and gradient).  This
///                     is expected to be a function object (e.g., a lambda or std::function) with
///                     the signature `Cost(const X_t& x, Gradient_t& grad)`, where `Cost` is the
///                     type of the cost function value, and `Gradient_t` is the type of the
///                     gradient vector (e.g.,  `std::vector<Scalar>` or an Eigen vector type).
///
/// @param x           A reference to the input variable. The gradient is checked at this point.
///                     The contents of x may be modified during the numerical gradient calculation.
/// @param acc         A function object (lambda, std::function, or functor) that calculates the
/// cost
///                     function value and its gradient.  It should have the signature
///                     `Cost(const X_t& x, Gradient_t& grad)` or
//                      `Cost(const X_t& x, Gradient_t& grad, Hessian_t &)`.  The function *must*
//                      compute
///                     the gradient and store it in the `grad` argument.
/// @param eps         The finite difference step size used for numerical gradient estimation.
///                     Defaults to 1e-2 for float and 1e-5 for double, based on the scalar type of
///                     X_t.
/// @param method      The method used for numerical gradient approximation.
/// @param verbose     If `true`, prints detailed information about the gradient check,
///                     including the numerical and analytical gradients.
///
/// @return            `true` if the analytical gradient (computed by 'acc') and the numerical
///                     gradient are within `eps` of each other; `false` otherwise.  Note: The
///                     current implementation always returns true.  You would need to add the
///                     actual comparison logic.
template <typename X_t, typename AccFunc>
inline bool CheckGradient(
    X_t &x, const AccFunc &acc,
    double eps = std::is_same_v<typename traits::params_trait<X_t>::Scalar, float> ? 1e-2 : 1e-5,
    const diff::Method method = diff::Method::kCentral, bool verbose = true) {
  using Scalar = traits::params_trait<X_t>::Scalar;
  constexpr int Dims = traits::params_trait<X_t>::Dims;
  using Gradient_t = Vector<Scalar, Dims>;
  using MaybeHessian_t = Matrix<Scalar, Dims, Dims>;
  constexpr bool GradIsScalar = std::is_scalar_v<Gradient_t>;
  constexpr bool HasH = !std::is_invocable_v<AccFunc, const X_t &, Gradient_t &>;
  // Get the gradient provided by 'acc'
  Gradient_t grad;
  if constexpr (GradIsScalar)
    grad = 0;
  else
    grad.setZero();

  auto fg = [&acc](const X_t &x, Gradient_t &g) {
    if constexpr (HasH &&
                  !std::is_invocable_v<AccFunc, const X_t &, Gradient_t &, MaybeHessian_t &>)
      return acc(x, g, SuperNul());
    else if constexpr (HasH) {
      MaybeHessian_t H = MaybeHessian_t::Zero();  // Support dense only for now
      return acc(x, g, H);
    } else
      return acc(x, g);
  };
  const auto res = fg(x, grad);

  // Estimate numerical jacobian
  auto f = [&acc](const X_t &x) {
    if constexpr (HasH)
      return acc(x, SuperNul(), SuperNul());
    else
      return acc(x, SuperNul());
  };
  const auto J_num = EstimateNumJac(x, f, method, eps / 10.0f);

  // Verify that grad = Jt * res
  Gradient_t grad_num;
  if constexpr (traits::is_pair_v<decltype(res)>)
    grad_num = J_num.transpose() * res.first;
  else
    grad_num = J_num.transpose() * res;

  if constexpr (GradIsScalar) {
    const auto diff = (grad - grad_num[0]);
    const auto max_diff = std::abs(diff);
    if (max_diff > eps) {
      if (verbose)
        TINYOPT_LOG("Wrong gradient {:.3e}>eps. \nGradient:{}, Numerial Gradient:{}", max_diff,
                    grad, grad_num[0]);
      return false;
    }
  } else {
    const auto diff = (grad - grad_num);
    const auto max_diff = diff.cwiseAbs().maxCoeff();
    if (max_diff > eps) {
      if (verbose)
        TINYOPT_LOG("Wrong gradient {:.3e}>eps. \nGradient:\n{}, \nNumerial Gradient:\n{}",
                    max_diff, grad, grad_num);
      return false;
    }
  }

  return true;
}

}  // namespace tinyopt::diff
