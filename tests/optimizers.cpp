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

void TestOptimizerSimple() {
  // Use Optimizer class interface
  {
    auto loss = [&](const auto &x, auto &grad, auto &H) {
      double res = x * x - 2, J = 2 * x;
      H(0, 0) = J * J;
      grad(0) = J * res;
      return res * res;
    };

    float x = 1;
    using Optimizer = Optimizer<SolverLM<Mat1f>>;
    Optimizer::Options options;
    options.log.print_x = true;
    Optimizer optimizer(options);
    const auto &out = optimizer(x, loss);
    REQUIRE(out.Succeeded());
    REQUIRE(out.Converged());
    REQUIRE(x == Approx(std::sqrt(2.0)).margin(1e-5));
  }
  // Use nlls::Optimize interface
  {
    auto loss = [&](const auto &x, auto &grad, auto &H) {
      double res = x * x - 2, J = 2 * x;
      H(0, 0) = J * J;
      grad(0) = J * res;
      return res * res;
    };

    float x = 1;
    const auto &out = lm::Optimize(x, loss);
    REQUIRE(out.Succeeded());
    REQUIRE(out.Converged());
    REQUIRE(x == Approx(std::sqrt(2.0)).margin(1e-5));
  }
}

void TestOptimizerAutoDiff() {
  // Use Optimizer class interface
  {
    auto loss = [&](const auto &x) {
      using T = typename std::remove_const_t<std::remove_reference_t<decltype(x)>>;
      return x * x - T(2.0);
    };

    float x = 1;
    using Optimizer = Optimizer<SolverLM<Vec1f>>;
    Optimizer::Options options;
    options.log.print_x = true;
    Optimizer optimizer(options);
    const auto &out = optimizer(x, loss);
    REQUIRE(out.Succeeded());
    REQUIRE(out.Converged());
    REQUIRE(x == Approx(std::sqrt(2.0)).margin(1e-5));
  }
  // Use nlls::Optimize interface
  {
    auto loss = [&](const auto &x) {
      using T = typename std::remove_const_t<std::remove_reference_t<decltype(x)>>;
      return x * x - T(2.0);
    };

    float x = 1;
    const auto &out = lm::Optimize(x, loss);
    REQUIRE(out.Succeeded());
    REQUIRE(out.Converged());
    REQUIRE(x == Approx(std::sqrt(2.0)).margin(1e-5));
  }
  {
    const Vec3 y_prior(3, 2, 1);

    Vec3 x = Vec3::Zero();

    auto loss = [&](const auto &x) { return (x - y_prior).eval(); };
    auto acc_loss = diff::NumDiff2(x, loss);

    if (1) {
      using Optimizer = Optimizer<SolverLM<Mat3>>;
      Optimizer::Options options;
      options.log.print_x = true;
      Optimizer optimizer(options);
      Optimizer::OutputType out;
      optimizer.Step(x, acc_loss, out);
      optimizer.Step(x, acc_loss, out);
      optimizer.Step(x, acc_loss, out);
      REQUIRE(out.Succeeded());
      REQUIRE(out.Converged());
    }
    if (1) {
      using Optimizer = Optimizer<SolverLM<Mat3>>;
      Optimizer optimizer;
      const auto &out = optimizer(x, acc_loss);
      REQUIRE(out.Succeeded());
      REQUIRE(out.Converged());
    }
  }
}

TEST_CASE("tinyopt_optimizer") {
  TestOptimizerSimple();
  TestOptimizerAutoDiff();
}