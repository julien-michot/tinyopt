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

#include <tinyopt/diff/num_diff.h>
#include <tinyopt/optimize.h>
#include <tinyopt/optimizers/optimizer.h>

using Catch::Approx;

using namespace tinyopt;
using namespace tinyopt::diff;
using namespace tinyopt::optimizers;
using namespace tinyopt::solvers;

/**
 * UNIT TEST: Beale Function
 * -------------------------------------------------------
 * Formula: f(x,y) = (1.5 - x + xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2
 * Difficulty: Has a sharp "flat" region at the corners.
 * Best starting point: (1, 1) or (0.1, 0.1).
 */
void test_beale_convergence() {
  TINYOPT_LOG("Beale");
  Vec2 x(1.0, 1.0);

  auto loss = [&](const auto &v) {
    using T = typename std::decay_t<decltype(v)>::Scalar;
    const T x_val = v(0), y_val = v(1);
    T t1 = T(1.5) - x_val + x_val * y_val;
    T t2 = T(2.25) - x_val + x_val * y_val * y_val;
    T t3 = T(2.625) - x_val + x_val * pow(y_val, 3);
    return Vector<T, 3>(t1, t2, t3);
  };

  using Optimizer = Optimizer<SolverLM<Mat2>>;
  Optimizer::Options options;
  options.log.print_x = true;
  options.max_iters = 200;
  options.max_consec_failures = 0;
  options.min_error = 1e-30;
  options.solver.damping_init = 1e-3;

  Optimizer optimizer(options);
  const auto &out = optimizer(x, loss);

  REQUIRE(out.num_diff_used == false);
  REQUIRE(out.Succeeded());
  // Minimum at (3, 0.5)
  REQUIRE(x(0) == Approx(3.0).margin(1e-4));
  REQUIRE(x(1) == Approx(0.5).margin(1e-4));
}

/**
 * UNIT TEST: Himmelblau's Function
 * -------------------------------------------------------
 * Formula: f(x,y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2
 * Difficulty: Has FOUR identical local minima.
 * This tests which minimum the optimizer "falls" into based on start.
 */
void test_himmelblau_convergence() {
  TINYOPT_LOG("Himmelblau");
  // Start near one of the four minima
  Vec2 x(3.5, 2.5);

  auto loss = [&](const auto &v) {
    using T = typename std::decay_t<decltype(v)>::Scalar;
    T x2 = v(0) * v(0);
    T y2 = v(1) * v(1);
    T t1 = x2 + v(1) - T(11.0);
    T t2 = v(0) + y2 - T(7.0);
    return Vector<T, 2>(t1, t2);
  };

  using Optimizer = Optimizer<SolverLM<Mat2>>;

  Optimizer::Options options;
  options.log.print_x = true;
  options.max_iters = 200;
  options.max_consec_failures = 0;
  options.min_error = 1e-30;
  options.solver.damping_init = 1e-4;

  Optimizer optimizer(options);
  const auto &out = optimizer(x, loss);

  REQUIRE(out.num_diff_used == false);
  // One of the solutions is (3.0, 2.0)
  REQUIRE(x(0) == Approx(3.0).margin(1e-4));
  REQUIRE(x(1) == Approx(2.0).margin(1e-4));
}

/**
 * UNIT TEST: Wood's Function (4D)
 * -------------------------------------------------------
 * A 4-variable problem that is notoriously difficult for
 * solvers that don't use high-precision Hessian info.
 */
void test_wood_convergence() {
  TINYOPT_LOG("Wood");
  // Standard starting point
  Vec4 x(-3.0, -1.0, -3.0, -1.0);

  auto loss = [&](const auto &v) {
    using T = typename std::decay_t<decltype(v)>::Scalar;
    const T x1 = v(0), x2 = v(1), x3 = v(2), x4 = v(3);

    T f1 = T(100.0) * pow(x2 - x1 * x1, 2);
    T f2 = pow(1.0 - x1, 2);
    T f3 = T(90.0) * pow(x4 - x3 * x3, 2);
    T f4 = pow(1.0 - x3, 2);
    T f5 = T(10.) * (pow(x2 - T(1.0), 2) + pow(x4 - T(1.0), 2));
    T f6 = T(19.8) * (x2 - T(1.0)) * (x4 - T(1.0));

    return Vector<T, 6>(f1, f2, f3, f4, f5, f6);
  };

  using Optimizer = Optimizer<SolverLM<Mat4>>;  // TODO use trust region instead
  Optimizer::Options options;
  options.log.print_x = true;
  options.max_iters = 500;  // Wood takes a while
  options.max_consec_failures = 0;
  options.min_error = 1e-30;
  options.min_rerr_dec = 0;
  options.solver.damping_init = 1e-2;

  Optimizer optimizer(options);
  const auto &out = optimizer(x, loss);

  REQUIRE(out.num_diff_used == false);
  REQUIRE(out.Succeeded());
  // Minimum at (1, 1, 1, 1)
  for (int i = 0; i < 4; ++i) REQUIRE(x(i) == Approx(1.0).margin(1e-3));
}

/**
 * UNIT TEST : Freudenstein and Roth Function
 * -------------------------------------------------------
 * Difficulty: Highly non-linear with severe local minima.
 * Often used to test if a solver can "escape" or handle
 * regions where the local quadratic approximation is very poor.
 */
void test_freudenstein_roth() {
  Vec2 x(0.5, -2.0);  // Standard starting point

  auto loss = [&](const auto &v, auto &grad, auto &H) {
    // Residuals: r1 = x1 - 13 + ((5 - x2)*x2 - 2)*x2
    //            r2 = x1 - 29 + ((x2 + 1)*x2 - 14)*x2
    double x1 = v(0), x2 = v(1);
    double r1 = x1 - 13.0 + ((5.0 - x2) * x2 - 2.0) * x2;
    double r2 = x1 - 29.0 + ((x2 + 1.0) * x2 - 14.0) * x2;

    if constexpr (!traits::is_nullptr_v<decltype(grad)>) {
        // First derivatives of residuals
        double dr1_dx2 = 10.0 * x2 - 3.0 * x2 * x2 - 2.0;
        double dr2_dx2 = 3.0 * x2 * x2 + 2.0 * x2 - 14.0;

        grad(0) = 2.0 * (r1 + r2);
        grad(1) = 2.0 * (r1 * dr1_dx2 + r2 * dr2_dx2);

        // Full Analytical Hessian H = 2 * (J^T * J + sum(r_i * d2r_i))
        // Second derivatives of residuals
        double d2r1_dx22 = 10.0 - 6.0 * x2;
        double d2r2_dx22 = 6.0 * x2 + 2.0;

        H(0, 0) = 4.0; // 2 * (dr1/dx1^2 + dr2/dx1^2) = 2 * (1 + 1)
        H(0, 1) = 2.0 * (dr1_dx2 + dr2_dx2);
        H(1, 0) = H(0, 1);
        H(1, 1) = 2.0 * (dr1_dx2 * dr1_dx2 + dr2_dx2 * dr2_dx2 + r1 * d2r1_dx22 + r2 * d2r2_dx22);
    }
    return r1 * r1 + r2 * r2;
  };

  // REQUIRE(CheckGradient(x, loss, 1e-10));

  // auto loss = [&](const auto &v) {
  //   using T = typename std::decay_t<decltype(v)>::Scalar;
  //   // Residuals: r1 = x1 - 13 + ((5 - x2)*x2 - 2)*x2
  //   //            r2 = x1 - 29 + ((x2 + 1)*x2 - 14)*x2
  //   T x1 = v(0), x2 = v(1);
  //   T r1 = x1 - 13.0 + ((5.0 - x2) * x2 - 2.0) * x2;
  //   T r2 = x1 - 29.0 + ((x2 + 1.0) * x2 - 14.0) * x2;
  //   return Vector<T, 2>(r1, r2);
  // };

  using Optimizer = Optimizer<SolverLM<Mat2>>;
  Optimizer::Options options;
  options.max_iters = 100;
  options.log.print_x = true;
  options.max_consec_failures = 0;
  options.min_error = 0;
  options.min_rerr_dec = 0;
  options.min_step_norm2 = 1e-36;
  options.min_grad_norm2 = 0;
  options.solver.damping_init = 1e-2;

  Optimizer optimizer(options);
  const auto &out = optimizer(x, loss);

  REQUIRE(out.Succeeded());
  // Global minimum is at (5, 4)
  REQUIRE(x(0) == Approx(5.0).margin(1e-4));
  REQUIRE(x(1) == Approx(4.0).margin(1e-4));
}

/**
 * UNIT TEST: Jennrich and Sampson Function
 * -------------------------------------------------------
 * Difficulty: Poorly conditioned.
 * This is a classic "small-residual" problem. The Jacobian
 * becomes nearly rank-deficient, testing the solver's
 * numerical precision and damping.
 */
void test_jennrich_sampson() {
  Vec2 x(0.3, 0.4);  // Standard start

  // auto loss = [&](const auto &v, auto &grad, auto &H) {
  //   double f = 0;
  //   if constexpr (!traits::is_nullptr_v<decltype(grad)>) {
  //     grad.setZero();
  //     H.setZero();
  //   }

  //   for (int i = 1; i <= 10; ++i) {
  //     double d_i = i;
  //     // r_i = 2 + 2i - (exp(i*x1) + exp(i*x2))
  //     double r = 2.0 + 2.0 * d_i - (std::exp(d_i * v(0)) + std::exp(d_i * v(1)));
  //     f += r * r;

  //     if constexpr (!traits::is_nullptr_v<decltype(grad)>) {
  //       double ji1 = -d_i * std::exp(d_i * v(0));
  //       double ji2 = -d_i * std::exp(d_i * v(1));

  //       grad(0) += 2.0 * r * ji1;
  //       grad(1) += 2.0 * r * ji2;

  //       H(0, 0) += ji1 * ji1;
  //       H(1, 1) += ji2 * ji2;
  //       H(0, 1) += ji1 * ji2;
  //     }
  //   }
  //   if constexpr (!traits::is_nullptr_v<decltype(H)>) H(1, 0) = H(0, 1);
  //   return f;
  // };

  // REQUIRE(CheckGradient(x, loss, 1e-10));

  auto loss = [&](const auto &v) {
    using T = typename std::decay_t<decltype(v)>::Scalar;
    Vector<T, 10> f;
    f.setZero();
    for (int i = 1; i <= 10; ++i) {
      T d_i(i);
      // r_i = 2 + 2i - (exp(i*x1) + exp(i*x2))
      f(i-1) = T(2.0) + T(2.0) * d_i - (exp(d_i * v(0)) + exp(d_i * v(1)));
    }
    return f;
  };


  using Optimizer = Optimizer<SolverLM<Mat2>>;
  Optimizer::Options options;
  options.max_iters = 500;
  options.log.print_x = true;
  options.max_consec_failures = 0;
  options.min_error = 1e-30;
  options.min_rerr_dec = 0;
  options.solver.damping_init = 1e-6;

  Optimizer optimizer(options);
  const auto &out = optimizer(x, loss);

  REQUIRE(out.Succeeded());
  // Minimum is at approx (0.2578, 0.2578)
  REQUIRE(x(0) == Approx(x(1)).margin(1e-5));
}

TEST_CASE("tinyopt_optimizer_nlls_hard") {
  test_beale_convergence();
  test_himmelblau_convergence();
  // test_wood_convergence(); -> TODO: use Trust Region
  // test_freudenstein_roth(); -> TODO: fix local minima
  test_jennrich_sampson();
}
