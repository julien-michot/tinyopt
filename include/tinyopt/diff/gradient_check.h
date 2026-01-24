// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tinyopt/diff/num_diff.h>
#include <tinyopt/math.h>

#include <tinyopt/log.h>
#include <tinyopt/traits.h>
#include <cstddef>
#include <type_traits>
#include "tinyopt/losses/norms.h"

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
  const int dims = traits::DynDims(x);
  using Gradient_t = Vector<Scalar, Dims>;
  constexpr bool HasH = !std::is_invocable_v<AccFunc, const X_t &, Gradient_t &>;
  constexpr bool DenseH =
      std::is_invocable_v<AccFunc, const X_t &, Gradient_t &, Matrix<Scalar, Dims, Dims> &>;
  using MaybeHessian_t = std::conditional_t<
      HasH, std::conditional_t<DenseH, Matrix<Scalar, Dims, Dims>, SparseMatrix<Scalar>>,
      std::nullptr_t>;

  std::nullptr_t nul;
  // Get the gradient provided by 'acc'
  Gradient_t grad(dims);
  grad.setZero();

  auto fg = [&](const X_t &x, Gradient_t &g) {
    if constexpr (HasH &&
                  !std::is_invocable_v<AccFunc, const X_t &, Gradient_t &, MaybeHessian_t &>)
      return acc(x, g, nul);
    else if constexpr (HasH) {
      MaybeHessian_t H;
      H.resize(dims, dims);
      H.setZero();
      return acc(x, g, H);
    } else
      return acc(x, g);
  };
  fg(x, grad);

  // Estimate numerical jacobian
  auto f = [&](const X_t &x) {
    if constexpr (HasH) {
      MaybeHessian_t H;
      H.resize(dims, dims);
      H.setZero();
      return acc(x, nul, H);
    } else
      return acc(x, nul);
  };

  const auto grad_num = EstimateNumJac(x, f, method, eps / 10.0f).transpose().eval();
  const double max_dist = (grad_num - grad).cwiseAbs().maxCoeff();
  const bool is_within_tol = max_dist < eps;
  if (!is_within_tol && verbose)
    TINYOPT_LOG("Wrong gradient {:.3e}>eps. \nGradient:{}, Numerial Gradient:{}", max_dist, grad,
                grad_num);
  return is_within_tol;
}

/// @brief Compares the gradient computed by the user-provided 'acc_res_func' function with a
///        numerically estimated gradient of the returned residuals's L2² norm.
///        It will also check the Hessian approximation as H ~ Jt*J if enabled.
//         This function helps verify the correctness of the
///        analytical gradient calculation in your cost/objective function.
///
/// @tparam X_t         Type of the input variable `x`.  This could be an Eigen matrix/vector type
///                     or a standard container like std::vector.  It must support basic arithmetic
///                     operations and element access.
/// @tparam AccWithResidualsFunc
///                     Type of the 'acc_res_func' function (which calculates the residuals and
///                     gradient). This is expected to be a function object (e.g., a lambda or
///                     std::function) with the signature `Residuals(const X_t& x, Gradient_t&
///                     grad)`, where `Residuals`is a scalar (one residual) or a Vector (multiple
///                     residuals), and `Gradient_t` is the type of the gradient vector.
///
/// @param x           A reference to the input variable. The gradient is checked at this point.
///                     The contents of x may be modified during the numerical gradient calculation.
/// @param acc_res_func An accumulation function that returns residuals instead of a cost
//                      (lambda, std::function, or functor) It should have one of the signatures
///                     `Residuals(const X_t& x, Gradient_t& grad)` or
//                      `Residuals(const X_t& x, Gradient_t& grad, Hessian_t &)`.  The function
//                      *must* compute the gradient and store it in the `grad` argument.
/// @param eps         The finite difference step size used for numerical gradient estimation.
///                     Defaults to 1e-2 for float and 1e-5 for double, based on the scalar type of
///                     X_t.
/// @param method      The method used for numerical gradient approximation.
/// @param verbose     If `true`, prints detailed information about the gradient check,
///                     including the numerical and analytical gradients.
/// @param check_H      Check that the Hessian is equal to Jt*J (only if the acc_res_func has a
/// 3rd argument)
/// @param downscale    Down scale the L2² norm by 2 or not.
///
/// @return            `true` if the analytical gradient (computed by 'residuals_func') and the
/// numerical
///                     gradient are within `eps` of each other; `false` otherwise.  Note: The
///                     current implementation always returns true.  You would need to add the
///                     actual comparison logic.
template <typename X_t, typename AccWithResidualsFunc>
inline bool CheckResidualsGradient(
    X_t &x, const AccWithResidualsFunc &acc_res_func,
    double eps = std::is_same_v<typename traits::params_trait<X_t>::Scalar, float> ? 1e-2 : 1e-5,
    const diff::Method method = diff::Method::kCentral, bool verbose = true, bool check_H = true,
    bool downscale = true) {
  using Scalar = traits::params_trait<X_t>::Scalar;
  constexpr int Dims = traits::params_trait<X_t>::Dims;
  const int dims = traits::DynDims(x);
  using Gradient_t = Vector<Scalar, Dims>;
  constexpr bool HasH = !std::is_invocable_v<AccWithResidualsFunc, const X_t &, std::nullptr_t &>;
  constexpr bool DenseH = std::is_invocable_v<AccWithResidualsFunc, const X_t &, Gradient_t &,
                                              Matrix<Scalar, Dims, Dims> &>;
  using MaybeHessian_t = std::conditional_t<
      HasH, std::conditional_t<DenseH, Matrix<Scalar, Dims, Dims>, SparseMatrix<Scalar>>,
      std::nullptr_t>;
  const double s = downscale ? 0.5 : 1.0;

  // Apply L2
  auto loss = [&](const X_t &x, auto &g, MaybeHessian_t &H) {
    if constexpr (HasH) {
      auto res = acc_res_func(x, g, H);

      using ResidualsType = std::decay_t<decltype(res)>;
      static_assert(!std::is_same_v<ResidualsType, Cost>,
                    "The function must return residuals, not a cost");
      return s * losses::SquaredL2(res);
    } else {
      auto res = acc_res_func(x, g);

      using ResidualsType = std::decay_t<decltype(res)>;
      static_assert(!std::is_same_v<ResidualsType, Cost>,
                    "The function must return residuals, not a cost");

      return s * losses::SquaredL2(res);
    }
  };
  bool success = CheckGradient(x, loss, eps, method, verbose);

  // Also check the hessian approx Jt*J
  if constexpr (HasH) {
    if (!check_H) return success;

    // Get user defined H
    Gradient_t g;
    g.resize(dims);
    g.setZero();
    MaybeHessian_t H;
    H.resize(dims, dims);
    H.setZero();
    acc_res_func(x, g, H);

    // Estimate numerical J and H = Jt*J
    auto f = [&](const X_t &x) {
      std::nullptr_t nul;
      return acc_res_func(x, nul, H);
    };

    const auto J_num = EstimateNumJac(x, f, method, eps / 10.0f);
    const auto H_num = (J_num.transpose() * J_num).eval();

    double max_dist;
    if constexpr (DenseH)
      max_dist = (H_num - H).cwiseAbs().maxCoeff();
    else {
      MaybeHessian_t H_num_sparse = H_num.sparseView();
      H_num_sparse.makeCompressed();
      max_dist = MaxAbsDiff(H_num_sparse, H);
    }

    const bool is_within_tol = max_dist < eps;
    if (!is_within_tol && verbose)
      TINYOPT_LOG("Wrong gradient {:.3e}>eps. \nH:\n{}, Numerial JtJ:\n{}", max_dist, H, H_num);
    return success && is_within_tol;
  }

  return success;
}

}  // namespace tinyopt::diff
