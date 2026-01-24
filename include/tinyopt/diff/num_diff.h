// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tinyopt/cost.h>
#include <tinyopt/math.h>  // Defines Matrix and Vector
#include <tinyopt/traits.h>
#include <cstddef>

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
   * offering improved performance in certain scenarios, e.g when using a
   * Manifold. In this case, accuracy will be traded with speed.
   *
   * Formula: (f(xh) - f(xh - 2 * h)) / (2 * h), with xh = x+h.
   */
  kFastCentral
};

/// Return the function `f` residuals with an estimate the jacobian d f(x)/d(x) around
/// `x` using numerical differentiation
template <typename X_t, typename Func>
auto NumEval(const X_t &x, const Func &f, const diff::Method &method = diff::Method::kCentral,
             typename traits::params_trait<X_t>::Scalar h =
                 FloatEpsilon<typename traits::params_trait<X_t>::Scalar>()) {
  using ptrait = traits::params_trait<X_t>;
  using Scalar = typename ptrait::Scalar;

  constexpr Index Dims = ptrait::Dims;
  const Index dims = traits::DynDims(x);

  // Support different function signatures
  auto fg = [&](const auto &x) {
    std::nullptr_t nul;
    if constexpr (std::is_invocable_v<Func, const X_t &>)
      return f(x);
    else if constexpr (std::is_invocable_v<Func, const X_t &, std::nullptr_t &>)
      return f(x, nul);
    else if constexpr (std::is_invocable_v<Func, const X_t &, std::nullptr_t &, std::nullptr_t &>)
      return f(x, nul, nul);
    else {  // likely a SparseMatrix<Scalar> hessian
      SparseMatrix<Scalar> H;
      H.resize(dims, dims);
      return f(x, nul, H);
    }
  };

  // Recover current residuals
  const auto res = fg(x);
  // Declare the jacobian matrix
  using ResType = typename std::decay_t<decltype(res)>;
  constexpr int ResDims = traits::params_trait<ResType>::Dims;
  const Index res_dims = traits::DynDims(res);

  using J_t = Matrix<Scalar, ResDims, Dims>;
  J_t J(res_dims, dims);

  // Estimate the jacobian using numerical differentiation
  Vector<Scalar, Dims> dx = Vector<Scalar, Dims>::Zero(dims);
  for (Index r = 0; r < dims; ++r) {
    X_t y = x;  // copy
    if (r > 0) dx[r - 1] = 0;
    dx[r] = h;
    ptrait::PlusEq(y, dx);
    const auto res_plus = fg(y);
    using ResType2 = typename std::decay_t<decltype(res_plus)>;
    if (method == Method::kCentral) {
      y = x;  // copy again
      dx[r] = -h;
      ptrait::PlusEq(y, dx);
      const auto res_minus = fg(y);
      if constexpr (std::is_scalar_v<ResType2>)
        J[r] = (res_plus - res_minus) / (2 * h);
      else
        J.col(r) = (res_plus.reshaped() - res_minus.reshaped()) / (2 * h);
    } else if (method == Method::kFastCentral) {
      dx[r] = -2 * h;  // given a small h, one can use this approximation, hopefully
      ptrait::PlusEq(y, dx);
      const auto res_minus = fg(y);
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
  return std::make_pair(res, J);
}

/// Estimate the jacobian of d f(x)/d(x) around `x` using numerical
/// differentiation
template <typename X_t, typename Func>
auto EstimateNumJac(const X_t &x, const Func &f,
                    const diff::Method &method = diff::Method::kCentral,
                    typename traits::params_trait<X_t>::Scalar h =
                        FloatEpsilon<typename traits::params_trait<X_t>::Scalar>()) {
  const auto &[res, J] = NumEval(x, f, method, h);
  return J;
}

/**
 * @brief Creates a numerical differentiation function for a given residuals
 * function.
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
 * @tparam Scalar        The scalar type of the residuals, Jacobian, and
 * Gradient.
 * @tparam Dims          The dimension of the input vector `x`.
 *
 * @param residuals     The residuals function to be differentiated.
 * @param method        The numerical differentiation method to use. Defaults to
 * `Method::kCentral`.
 * @param h             The delta added to 'x' to compute the difference on each
 * dimension
 *
 * @return              A `std::function` object that takes an input `x`, a
 * vector for the residuals, and a matrix for the Jacobian as arguments.
 *
 * @note                The `Method` enum should be defined elsewhere and
 * include values like `Method::kForward`, `Method::kBackward`,
 * `Method::kCentral` and `Method::kFastCentral`.
 *
 * @note                The step size used for numerical differentiation is
 * determined internally and may be adapted based on the
 * magnitude of the input `x` to avoid numerical instability.
 *
 * @note                The generated `std::function` does not modify the input
 * `x`.
 *
 * @code
 *
 * // Example residuals function
 * auto loss = [](const std::vector<double>& x) {
 * return x[0] * x[0] + x[1];
 * };
 *
 * std::vector<double> x = {1.0, 2.0};
 * auto acc_loss = CreateNumDiffFunc1(x, loss, Method::kCentral);
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
auto CreateNumDiffFunc1(X_t &, const ResidualsFunc &residuals,
                        const Method &method = Method::kCentral,
                        typename traits::params_trait<X_t>::Scalar h =
                            FloatEpsilon<typename traits::params_trait<X_t>::Scalar>()) {
  auto loss = [&residuals, method, h](const auto &x, auto &grad) {
    constexpr bool HasGrad = !traits::is_nullptr_v<decltype(grad)>;
    // Recover current residuals
    const auto res = residuals(x);

    if constexpr (HasGrad) {
      const auto J = EstimateNumJac(x, residuals, method, h);
      grad = J.transpose() * res;
    }

    using ResType = typename std::decay_t<decltype(res)>;
    if constexpr (std::is_scalar_v<ResType>) {
      return std::abs(res);
    } else {
      // Returns the norm + number of residuals
      return Cost(res.matrix().norm(), res.size());
    }
  };
  return loss;
}

/**
 * @brief Creates a numerical differentiation function for a given residuals
 * function.
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
 * @tparam Scalar        The scalar type of the residuals, Jacobian, and
 * Hessian.
 * @tparam Dims          The dimension of the input vector `x`.
 *
 * @param residuals     The residuals function to be differentiated.
 * @param method        The numerical differentiation method to use. Defaults to
 * `Method::kCentral`.
 * @param h             The delta added to 'x' to compute the difference on each
 * dimension.
 *
 * @return              A `std::function` object that takes an input `x`, a
 * vector for the residuals, and a matrix for the Jacobian as arguments. It also
 * takes an optional matrix for the Hessian as an argument.
 *
 * @note                The `Method` enum should be defined elsewhere and
 * include values like `Method::kCentral`, `Method::kBackward`,
 * `Method::kCentral` and `Method::kFastCentral`.
 *
 * @note                The step size used for numerical differentiation is
 * determined internally and may be adapted based on the
 * magnitude of the input `x` to avoid numerical instability.
 *
 * @note                The generated `std::function` does not modify the input
 * `x`.
 *
 * @code
 *
 * // Example residuals function
 * auto loss = [](const std::vector<double>& x) {
 * return x[0] * x[0] + x[1];
 * };
 *
 * std::vector<double> x = {1.0, 2.0};
 * auto acc_loss = CreateNumDiffFunc1(x, loss, Method::kCentral);
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
auto CreateNumDiffFunc2(X_t &, const ResidualsFunc &residuals,
                        const Method &method = Method::kCentral,
                        typename traits::params_trait<X_t>::Scalar h =
                            FloatEpsilon<typename traits::params_trait<X_t>::Scalar>()) {
  auto loss = [&residuals, method, h](const auto &x, auto &grad, auto &H) {
    constexpr bool HasGrad = !traits::is_nullptr_v<decltype(grad)>;
    constexpr bool HasH = !traits::is_nullptr_v<decltype(H)>;
    // Recover current residuals
    const auto res = residuals(x);

    if constexpr (HasGrad) {
      const auto J = EstimateNumJac(x, residuals, method, h);
      grad = J.transpose() * res;
      if constexpr (HasH) H = J.transpose() * J;
    }

    using ResType = typename std::decay_t<decltype(res)>;
    if constexpr (std::is_scalar_v<ResType>) {
      return std::abs(res);
    } else {
      // Returns the norm + number of residuals
      return Cost(res.matrix().norm(), res.size());
    }
  };
  return loss;
}

}  // namespace tinyopt::diff
