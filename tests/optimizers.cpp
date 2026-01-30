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
using namespace tinyopt::solvers;

TEST_CASE("tinyopt_optimizer_converge") {
  // Use Optimizer class interface
  {
    auto loss = [&](const auto &x, auto &grad, auto &H) {
      double res = x * x - 2, J = 2 * x;
      if constexpr (!traits::is_nullptr_v<decltype(grad)>) {
        grad(0) = J * res;
        H(0, 0) = J * J;
      }
      using std::abs;
      return abs(res);
    };

    float x = 1;
    using Optimizer = Optimizer_<SolverLM<Mat1f>>;
    Options options;
    options.log.print_x = true;
    Optimizer optimizer(options);
    const auto &out = optimizer(x, loss);
    REQUIRE(out.Succeeded());
    REQUIRE(out.Converged());
    REQUIRE(x == Approx(std::sqrt(2.0)).margin(1e-5));
  }
  // Use Optimize interface
  {
    auto loss = [&](const auto &x, auto &grad, auto &H) {
      float res = x * x - 2, J = 2 * x;
      if constexpr (!traits::is_nullptr_v<decltype(grad)>) {
        grad(0) = J * res;
        H(0, 0) = J * J;
      }
      return std::abs(res);
    };

    float x = 1;
    const auto &out = Optimize(x, loss);
    REQUIRE(out.Succeeded());
    REQUIRE(out.Converged());
    REQUIRE(x == Approx(std::sqrt(2.0)).margin(1e-5));
  }
}

TEST_CASE("tinyopt_optimizer_autodiff") {
  // Use Optimizer class interface
  {
    auto loss = [&](const auto &x) {
      using T = typename std::decay_t<decltype(x)>;
      return x * x - T(2.0);
    };

    float x = 1;
    using Optimizer = Optimizer_<SolverLM<Vec1f>>;
    Options options;
    options.log.print_x = true;
    Optimizer optimizer(options);
    const auto &out = optimizer(x, loss);
    REQUIRE(out.Succeeded());
    REQUIRE(out.Converged());
    REQUIRE(x == Approx(std::sqrt(2.0)).margin(1e-5));
  }
  // Use Optimize interface
  {
    auto loss = [&](const auto &x) {
      using T = typename std::decay_t<decltype(x)>;
      return x * x - T(2.0);
    };

    float x = 1;
    const auto &out = Optimize(x, loss);
    REQUIRE(out.Succeeded());
    REQUIRE(out.Converged());
    REQUIRE(x == Approx(std::sqrt(2.0)).margin(1e-5));
  }
  {
    const Vec3 y_prior(3, 2, 1);

    Vec3 x = Vec3::Zero();

    auto loss = [&](const auto &x) { return (x - y_prior).eval(); };
    auto acc_loss = diff::CreateNumDiffFunc2(x, loss);

    if (1) {
      using Optimizer = Optimizer_<SolverLM<Mat3>>;
      Optimizer optimizer;
      const auto &out = optimizer(x, acc_loss);
      REQUIRE(out.Succeeded());
      REQUIRE(out.Converged());
    }
  }
}
