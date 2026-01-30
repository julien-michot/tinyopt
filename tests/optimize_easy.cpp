// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <cmath>

#include <Eigen/Eigen>

#if CATCH2_VERSION == 2
#include <catch2/catch.hpp>
#else
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#endif

#include <tinyopt/tinyopt.h>

#include <tinyopt/diff/gradient_check.h>
#include <tinyopt/diff/num_diff.h>
#include <tinyopt/optimize.h>
#include <tinyopt/optimizers/optimizer.h>

using Catch::Approx;

using namespace tinyopt;
using namespace tinyopt::diff;
using namespace tinyopt::solvers;

/**
 * UNIT TEST: Rosenbrock Function (The "Banana" Function)
 * -------------------------------------------------------
 * Formula: f(x, y) = (a - x)^2 + b(y - x^2)^2
 * Global minimum at (a, a^2). Usually a=1, b=100.
 * Difficulty: Narrow, curved valley that is easy to find but hard to converge to the minimum.
 */
void test_rosenbrock_convergence() {
  TINYOPT_LOG("ROSENBROCK");
  // Starting point
  Vec2 x(-1.2, 1.0);

  auto loss = [&](const auto &v, auto &grad, auto &H) {
    double x_val = v(0);
    double y_val = v(1);

    double term1 = 1.0 - x_val;
    double term2 = y_val - x_val * x_val;

    if constexpr (!traits::is_nullptr_v<decltype(grad)>) {
      // First derivatives
      grad(0) = -2.0 * term1 - 400.0 * x_val * term2;
      grad(1) = 200.0 * term2;

      // Hessian (Second derivatives)
      H(0, 0) = 2.0 - 400.0 * y_val + 1200.0 * x_val * x_val;
      H(0, 1) = -400.0 * x_val;
      H(1, 0) = -400.0 * x_val;
      H(1, 1) = 200.0;
    }

    return term1 * term1 + 100.0 * term2 * term2;
  };

  REQUIRE(CheckGradient(x, loss, 1e-5));

  using Optimizer = Optimizer_<SolverLM<Mat2>>;
  Optimizer::Options options;
  options.log.print_x = true;
  options.max_iters = 200;
  options.min_rerr_dec = 0;
  options.max_consec_failures = 20;

  Optimizer optimizer(options);
  const auto &out = optimizer(x, loss);

  REQUIRE(out.Succeeded());
  REQUIRE(out.Converged());
  // Minimum should be at (1, 1)
  REQUIRE(x(0) == Approx(1.0).margin(1e-5));
  REQUIRE(x(1) == Approx(1.0).margin(1e-5));
}

/**
 * UNIT TEST: Plateau Function (Easom-like Flat Surface)
 * -------------------------------------------------------
 * Formula: f(x, y) = -cos(x)cos(y)exp(-((x-pi)^2 + (y-pi)^2))
 * Difficulty: The function is nearly zero (flat plateau) everywhere except near the minimum.
 * Converging here requires the optimizer to handle very small gradients.
 */
void test_plateau_convergence() {
  TINYOPT_LOG("PLATEAU");
  const double PI = std::acos(-1.0);
  Vec2 x(3.0, 3.0);  // Start close to the dip

  auto loss = [&](const auto &v, auto &grad, auto &H) {
    double dx = v(0) - PI;
    double dy = v(1) - PI;
    double ex = exp(-(dx * dx + dy * dy));
    double cx = cos(v(0));
    double cy = cos(v(1));
    double sx = sin(v(0));
    double sy = sin(v(1));

    double cost = 1.0 - (cx * cy * ex);

    if constexpr (!traits::is_nullptr_v<decltype(grad)>) {
      // Gradient of cost:
      // d/dx [-cx * cy * ex] = -[(-sx)*cy*ex + cx*cy*ex*(-2*dx)]
      //                      = cy*ex*(sx + 2*dx*cx)
      double g0 = cy * ex * (sx + 2.0 * dx * cx);
      double g1 = cx * ex * (sy + 2.0 * dy * cy);

      grad(0) = g0;
      grad(1) = g1;

      // For Levenberg-Marquardt, we want the Hessian of a sum of squares.
      // If cost = r^2, then H approx 2 * J^T * J.
      // Since we are returning the total cost directly, the Hessian of 'cost'
      // should be provided. For the Easom function, the curvature is very low
      // on the plateau, so the full analytical Hessian is preferred.

      H(0, 0) = cy * ex * (cx - 4.0 * dx * sx + (2.0 - 4.0 * dx * dx) * cx);
      H(1, 1) = cx * ex * (cy - 4.0 * dy * sy + (2.0 - 4.0 * dy * dy) * cy);
      H(0, 1) = ex * (sx + 2.0 * dx * cx) * (sy + 2.0 * dy * cy);
      H(1, 0) = H(0, 1);
    }

    return cost;
  };

  REQUIRE(CheckGradient(x, loss, 1e-5));

  using Optimizer = Optimizer_<SolverLM<Mat2>>;
  Optimizer::Options options;
  options.lm.damping_init = 1e-6;
  options.log.print_x = true;
  // options.min_error = 0;

  Optimizer optimizer(options);
  const auto &out = optimizer(x, loss);

  REQUIRE(out.Succeeded());
  // Global minimum at (PI, PI)
  REQUIRE(x(0) == Approx(PI).margin(1e-4));
  REQUIRE(x(1) == Approx(PI).margin(1e-4));
}

/**
 * UNIT TEST: Powell Singular Function
 * -------------------------------------------------------
 * Formula: f = (x1 + 10x2)^2 + 5(x3 - x4)^2 + (x2 - 2x3)^4 + 10(x1 - x4)^4
 * Difficulty: The Hessian is singular at the solution (0,0,0,0).
 * Tests the optimizer's ability to handle ill-conditioned matrices.
 */
void test_powell_singular_convergence() {
  TINYOPT_LOG("POWELL");
  // Starting point
  Vec4 x(3.0, -1.0, 0.0, 1.0);

  auto loss = [&](const auto &v, auto &grad, auto &H) {
    double x1 = v(0), x2 = v(1), x3 = v(2), x4 = v(3);

    double t1 = x1 + 10.0 * x2;
    double t2 = x3 - x4;
    double t3 = x2 - 2.0 * x3;
    double t4 = x1 - x4;

    if constexpr (!traits::is_nullptr_v<decltype(grad)>) {
      grad.setZero();
      grad(0) = 2.0 * t1 + 40.0 * std::pow(t4, 3);
      grad(1) = 20.0 * t1 + 4.0 * std::pow(t3, 3);
      grad(2) = 10.0 * t2 - 8.0 * std::pow(t3, 3);
      grad(3) = -10.0 * t2 - 40.0 * std::pow(t4, 3);

      // Full Analytical Hessian -> JtJ approx is too slow to converge
      H.setZero();
      // Second derivatives of (x1 + 10x2)^2
      H(0, 0) = 2.0;
      H(0, 1) = 20.0;
      H(1, 0) = 20.0;
      H(1, 1) = 200.0;
      // Second derivatives of 5(x3 - x4)^2
      H(2, 2) += 10.0;
      H(2, 3) += -10.0;
      H(3, 2) += -10.0;
      H(3, 3) += 10.0;
      // Second derivatives of (x2 - 2x3)^4
      double d3 = 12.0 * t3 * t3;
      H(1, 1) += d3;
      H(1, 2) += -2.0 * d3;
      H(2, 1) += -2.0 * d3;
      H(2, 2) += 4.0 * d3;
      // Second derivatives of 10(x1 - x4)^4
      double d4 = 120.0 * t4 * t4;
      H(0, 0) += d4;
      H(0, 3) += -d4;
      H(3, 0) += -d4;
      H(3, 3) += d4;
    }

    return t1 * t1 + 5.0 * t2 * t2 + std::pow(t3, 4) + std::pow(t4, 4) * 10.0;
  };

  REQUIRE(CheckGradient(x, loss, 1e-5));

  using Optimizer = Optimizer_<SolverLM<Mat4>>;
  Optimizer::Options options;
  options.max_iters = 200;
  options.max_consec_failures = 0;
  options.min_error = 1e-30;
  options.min_rerr_dec = 1e-30;
  options.log.print_x = true;
  options.lm.damping_init = 1e-1;

  Optimizer optimizer(options);
  const auto &out = optimizer(x, loss);

  REQUIRE(out.Succeeded());
  // Minimum should be at (0, 0, 0, 0)
  for (int i = 0; i < 4; ++i) {
    REQUIRE(std::abs(x(i)) < 1e-3);
  }
}

TEST_CASE("tinyopt_optimizer_nlls_easy") {
  test_rosenbrock_convergence();
  test_plateau_convergence();
  test_powell_singular_convergence();
}
