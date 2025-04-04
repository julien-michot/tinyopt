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

#include <iostream>

#include <tinyopt/math.h>  // Defines Matrix and Vector
#include <tinyopt/traits.h>

namespace tinyopt::diff {
/**
 * @enum Method
 * @brief Specifies the method used for numerical differentiation.
 *
 * This enumeration controls the type of numerical differentiation performed.
 * Numerical differentiation is used to approximate the derivative of a function
 * at a given point by using finite differences.
 */
enum Method {
  /**
   * @brief Forward difference method.
   *
   * Approximates the derivative using the function values at the current point
   * and a point slightly ahead. It is a first-order approximation.
   *
   * Formula: (f(x + h) - f(x)) / h
   */
  kForward = 0,

  /**
   * @brief Central difference method.
   *
   * Approximates the derivative using the function values at points slightly
   * before and after the current point. It is a second-order approximation,
   * generally more accurate than the forward difference method.
   *
   * Formula: (f(x + h) - f(x - h)) / (2 * h)
   */
  kCentral,

  /**
   * @brief Fast central difference method.
   *
   * An optimized variant of the central difference method, potentially
   * offering improved performance in certain scenarios, e.g when using a Manifold.
   * In this case, accuracy will be traded with speed.
   *
   * Formula: (f(xh) - f(xh - 2 * h)) / (2 * h), with xh = x+h.
   */
  kFastCentral
};

/// Estimate the jacobian of d f(x)/d(x) around `x` using numerical differentiation
template <typename X_t, typename Func>
auto EstimateNumJac(const X_t &x, const Func &f,
                    const diff::Method &method = diff::Method::kCentral,
                    typename traits::params_trait<X_t>::Scalar h =
                        FloatEpsilon<typename traits::params_trait<X_t>::Scalar>()) {
  using ptrait = traits::params_trait<X_t>;
  using Scalar = typename ptrait::Scalar;
  constexpr int Dims = ptrait::Dims;

  int dims = Dims;
  if constexpr (Dims == Dynamic) dims = ptrait::dims(x);
  // Recover current residuals
  const auto res = f(x);
  // Declare the jacobian matrix
  using ResType = typename std::remove_const_t<std::remove_reference_t<decltype(res)>>;
  constexpr int ResDims = traits::params_trait<ResType>::Dims;
  int res_dims = ResDims;
  if constexpr (ResDims == Dynamic) res_dims = traits::params_trait<ResType>::dims(res);

  using J_t = Matrix<Scalar, ResDims, Dims>;
  J_t J(res_dims, dims);

  // Estimate the jacobian using numerical differentiation
  Vector<Scalar, Dims> dx = Vector<Scalar, Dims>::Zero(dims);
  for (int r = 0; r < dims; ++r) {
    X_t y = x;  // copy
    if (r > 0) dx[r - 1] = 0;
    dx[r] = h;
    ptrait::pluseq(y, dx);
    const auto res_plus = f(y);
    using ResType2 = typename std::remove_reference_t<std::remove_const_t<decltype(res_plus)>>;
    if (method == Method::kCentral) {
      y = x;  // copy again
      dx[r] = -h;
      ptrait::pluseq(y, dx);
      const auto res_minus = f(y);
      if constexpr (std::is_scalar_v<ResType2>)
        J[r] = (res_plus - res_minus) / (2 * h);
      else
        J.col(r) = (res_plus.reshaped() - res_minus.reshaped()) / (2 * h);
    } else if (method == Method::kFastCentral) {
      dx[r] = -2 * h;  // given a small h, one can use this approximation, hopefully
      ptrait::pluseq(y, dx);
      const auto res_minus = f(y);
      if constexpr (std::is_scalar_v<ResType2>)
        J[r] = (res_plus - res_minus) / (2 * h);
      else
        J.col(r) = (res_plus.reshaped() - res_minus.reshaped()) / (2 * h);
    } else {
      if constexpr (std::is_scalar_v<ResType2>)
        J[r] = (res_plus - res) / h;
      else
        J.col(r) = (res_plus.reshaped() - res) / h;
    }
  }
  return J;
}

/**
 * @brief Creates a numerical differentiation function for a given residuals function.
 *
 * This function generates a callable object (std::function) that, when invoked,
 * calculates the residuals and gradient of the provided
 * `residuals` function at a given input `x`. It offers different numerical
 * differentiation methods, such as forward, backward, and central differences.
 *
 * @tparam X_t           The type of the input vector `x`. It should support
 * arithmetic operations and element-wise access.
 * @tparam ResidualsFunc The type of the residuals function. It should be a
 * callable object that takes an `X_t` and returns a
 * scalar value.
 * @tparam Scalar        The scalar type of the residuals, Jacobian, and Gradient.
 * @tparam Dims          The dimension of the input vector `x`.
 *
 * @param residuals     The residuals function to be differentiated.
 * @param method        The numerical differentiation method to use. Defaults to
 * `Method::kCentral`.
 * @param h             The delta added to 'x' to compute the difference on each dimension
 *
 * @return              A `std::function` object that takes an input `x`, a vector for
 * the residuals, and a matrix for the Jacobian as arguments.
 * The `std::function` returns a `Scalar` value.
 * The function signature is:
 * `std::function<Scalar(const X_t &, Vector<Scalar, Dims> &)>`.
 *
 * @note                The `Method` enum should be defined elsewhere and include
 * values like `Method::kForward`, `Method::kBackward`, `Method::kCentral`
 * and `Method::kFastCentral`.
 *
 * @note                The step size used for numerical differentiation is
 * determined internally and may be adapted based on the
 * magnitude of the input `x` to avoid numerical instability.
 *
 * @note                The generated `std::function` does not modify the input `x`.
 *
 * @code
 *
 * // Example residuals function
 * auto loss = [](const std::vector<double>& x) {
 * return x[0] * x[0] + x[1];
 * };
 *
 * std::vector<double> x = {1.0, 2.0};
 * auto acc_loss = NumDiff1(x, loss, Method::kCentral);
 *
 * Eigen::Vector2d grad;
 *
 * double norm = acc_loss(x, g, H);
 *
 * The returned function can be passed to an optimizer, e.g.
 * auto optimizer = Optimizer<SolverGD<Vec2>>();
 * optimizer(x, acc_loss);
 *
 * @endcode
 */
template <typename X_t, typename ResidualsFunc>
auto NumDiff1(X_t &, const ResidualsFunc &residuals, const Method &method = Method::kCentral,
              typename traits::params_trait<X_t>::Scalar h =
                  FloatEpsilon<typename traits::params_trait<X_t>::Scalar>()) {
  using ptrait = traits::params_trait<X_t>;
  using Scalar = typename ptrait::Scalar;
  constexpr int Dims = ptrait::Dims;

  using Func = std::function<Scalar(const X_t &, Vector<Scalar, Dims> &)>;

  Func loss = [&residuals, method, h](const auto &x, auto &grad) {
    // Declare the jacobian matrix
    const auto J = EstimateNumJac(x, residuals, method, h);
    // Recover current residuals
    const auto res = residuals(x);
    using ResType = typename std::remove_const_t<std::remove_reference_t<decltype(res)>>;
    if constexpr (std::is_scalar_v<ResType>) {
      grad = J.transpose() * res;  // TODO speed this up by avoiding to store J
      return std::abs(res);
    } else {
      grad = J.transpose() * res;
      return res.norm();
    }
  };
  return loss;
}

/**
 * @brief Creates a numerical differentiation function for a given residuals function.
 *
 * This function generates a callable object (std::function) that, when invoked,
 * calculates the residuals, gradient and Hessian of the provided
 * `residuals` function at a given input `x`. It offers different numerical
 * differentiation methods, such as forward, backward, and central differences.
 *
 * @tparam X_t           The type of the input vector `x`. It should support
 * arithmetic operations and element-wise access.
 * @tparam ResidualsFunc The type of the residuals function. It should be a
 * callable object that takes an `X_t` and returns a
 * scalar value.
 * @tparam Scalar        The scalar type of the residuals, Jacobian, and Hessian.
 * @tparam Dims          The dimension of the input vector `x`.
 *
 * @param residuals     The residuals function to be differentiated.
 * @param method        The numerical differentiation method to use. Defaults to
 * `Method::kCentral`.
 * @param h             The delta added to 'x' to compute the difference on each dimension.
 *
 * @return              A `std::function` object that takes an input `x`, a vector for
 * the residuals, and a matrix for the Jacobian as arguments.
 * It also takes an optional matrix for the Hessian as an argument.
 * The `std::function` returns a `Scalar` value.
 * The function signature is:
 * `std::function<Scalar(const X_t &, Vector<Scalar, Dims> &, Matrix<Scalar, Dims, Dims> &)>`.
 *
 * @note                The `Method` enum should be defined elsewhere and include
 * values like `Method::kCentral`, `Method::kBackward`, `Method::kCentral`
 * and `Method::kFastCentral`.
 *
 * @note                The step size used for numerical differentiation is
 * determined internally and may be adapted based on the
 * magnitude of the input `x` to avoid numerical instability.
 *
 * @note                The generated `std::function` does not modify the input `x`.
 *
 * @code
 *
 * // Example residuals function
 * auto loss = [](const std::vector<double>& x) {
 * return x[0] * x[0] + x[1];
 * };
 *
 * std::vector<double> x = {1.0, 2.0};
 * auto acc_loss = NumDiff1(x, loss, Method::kCentral);
 *
 * Eigen::Vector2d grad;
 * Eigen::Matrix2d H;
 *
 * double norm = acc_loss(x, g, H);
 *
 * The returned function can be passed to an optimizer, e.g.
 * auto optimizer = Optimizer<SolverLM<Mat2>>();
 * optimizer(x, acc_loss);
 *
 * @endcode
 */
template <typename X_t, typename ResidualsFunc>
auto NumDiff2(X_t &, const ResidualsFunc &residuals, const Method &method = Method::kCentral,
              typename traits::params_trait<X_t>::Scalar h =
                  FloatEpsilon<typename traits::params_trait<X_t>::Scalar>()) {
  using ptrait = traits::params_trait<X_t>;
  using Scalar = typename ptrait::Scalar;
  constexpr int Dims = ptrait::Dims;

  using Func =
      std::function<Scalar(const X_t &, Vector<Scalar, Dims> &, Matrix<Scalar, Dims, Dims> &)>;

  Func loss = [&residuals, method, h](const auto &x, auto &grad, auto &H) {
    // Declare the jacobian matrix
    const auto J = EstimateNumJac(x, residuals, method, h);
    // Recover current residuals
    const auto res = residuals(x);
    using ResType = typename std::remove_const_t<std::remove_reference_t<decltype(res)>>;
    if constexpr (std::is_scalar_v<ResType>) {
      grad = J.transpose() * res;
      H = J.transpose() * J;
      return std::abs(res);
    } else {
      grad = J.transpose() * res;
      H = J.transpose() * J;
      return res.cwiseAbs().sum();
    }
  };
  return loss;
}

}  // namespace tinyopt::diff
