// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <Eigen/Eigen>
#include <cmath>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <tinyopt/diff/gradient_check.h>
#include <tinyopt/diff/jet.h>
#include <tinyopt/optimize.h>
#include <tinyopt/optimizers/lm.h>
#include <tinyopt/tinyopt.h>

using Catch::Approx;
using namespace tinyopt;

namespace {

// Helper: Standard test options
inline Options CreateDefaultTestOptions() {
  Options options;
  options.stop.max_iters = 50;
  options.stop.max_consec_failures = 0;
  options.log.enable = false;  // Set to true when debugging step-by-step
  return options;
}

/**
 * Example 1: Analytical Residual & Derivatives
 * ---------------------------------------------
 * Target: minimize r(x) = x^2 - target
 * Analytical Jacobian: J = dr/dx = 2*x
 * Linear System Accumulation:
 *   grad += J^T * r
 *   H    += J^T * J
 */
void TestAnalyticalResiduals(double target) {
  auto residuals = [target](const auto &x, auto &grad, auto &H) {
    double res = x * x - target;
    double J = 2.0 * x;

    if constexpr (!traits::is_nullptr_v<decltype(grad)>) {
      grad(0) = J * res;
      H(0, 0) = J * J;
    }
    return res;
  };

  auto loss = [&](const auto &x, auto &grad, auto &H) {
    double r = residuals(x, grad, H);
    return r * r;
  };

  double x0 = 1.0;

  // 1. MANDATORY: Verify gradient matches numerical approximation
  REQUIRE(diff::CheckResidualsGradient(x0, residuals));

  // 2. Run optimization
  Options options = CreateDefaultTestOptions();
  const auto &out = Optimize(x0, loss, options);

  // 3. Verify convergence and mathematical optimality
  REQUIRE(out.Succeeded());
  REQUIRE(out.Converged());
  REQUIRE(std::abs(x0) == Approx(std::sqrt(target)).margin(1e-5));
}

/**
 * Example 2: Automatic Differentiation via Jets (Dual Numbers)
 * ------------------------------------------------------------
 * No manual derivatives required!
 */
void TestAutoDiffJet(double target) {
  auto loss = [target](const auto &x) { return x * x - target; };

  double x = 1.5;
  Options options = CreateDefaultTestOptions();
  options.cost.use_squared_norm = true;
  options.cost.downscale_by_2 = true;

  const auto &out = Optimize(x, loss, options);

  REQUIRE(out.Succeeded());
  REQUIRE(out.Converged());
  REQUIRE(std::abs(x) == Approx(std::sqrt(target)).margin(1e-5));
}

}  // namespace

TEST_CASE("tinyopt_template_verification", "[optimization]") {
  // Test across multiple starting values using Catch2 generators
  auto target = GENERATE(2.0, 3.0, 9.0, 16.0);
  CAPTURE(target);

  SECTION("Analytical Residuals with Gradient Checking") { TestAnalyticalResiduals(target); }

  SECTION("Jet Automatic Differentiation") { TestAutoDiffJet(target); }
}
