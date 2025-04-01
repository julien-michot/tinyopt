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

#include <chrono>
#include <cmath>
#include <thread>

#if CATCH2_VERSION == 2
#include <catch2/catch.hpp>
#else
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#endif

#include <tinyopt/tinyopt.h>

using namespace tinyopt;
using namespace tinyopt::lm;

/// Common checks on an successful optimization
void SuccessChecks(const auto &out, int min_num_iters = 1,
                   StopReason expected_stop = StopReason::kMinGradNorm) {
  REQUIRE(out.Succeeded());
  REQUIRE(out.num_iters >= min_num_iters);
  if (min_num_iters > 0) {
    REQUIRE(out.last_err2 < 1e-5);
    REQUIRE(out.Converged());
    REQUIRE(out.errs2.size() == size_t(out.num_iters + 1));
    REQUIRE(out.successes.size() == size_t(out.num_iters + 1));
    REQUIRE(out.deltas2.size() == size_t(out.num_iters + 1));
  }
  REQUIRE(out.last_H(0, 0) > 0);  // was exported
  std::cout << out.StopReasonDescription() << "\n";
  REQUIRE(out.stop_reason == expected_stop);
}

void TestSuccess() {
  // Normal case using LM
  {
    std::cout << "**** Normal Test Case LM \n";
    auto loss = [&](const auto &x, auto &grad, auto &H) {
      double res = x - 2;
      H(0, 0) = 1;
      grad(0) = res;
      return res * res;
    };
    double x = 1;
    const auto &out = lm::Optimize(x, loss);
    SuccessChecks(out);
  }
  {
    std::cout << "**** min || ||x-y||² || \n";
    const Vec2 y = 10 * Vec2::Random();  // prior
    auto loss = [&](const auto &x) {
      const auto res = (x - y).eval();
      return res.squaredNorm();  // return the sum
    };

    Vec2 x(5, 5);
    lm::Options options;
    options.solver.damping_init = 1e0;
    options.log.print_rmse = true;
    const auto &out = lm::Optimize(x, loss, options);
    REQUIRE(out.Succeeded());
    REQUIRE(!out.Converged());
  }
  // Normal case using LM
  {
    std::cout << "**** Normal Test Case GN\n";
    auto loss = [&](const auto &x, auto &grad, auto &H) {
      double res = x - 2;
      H(0, 0) = 1;
      grad(0) = res;
      return res * res;
    };
    double x = 1;
    const auto &out = gn::Optimize(x, loss);
    SuccessChecks(out);
  }
  // Timimg out
  {
    std::cout << "**** Testing Time out x\n";
    auto loss = [&](const auto &x, auto &grad, auto &H) {
      double res = x - VecXf::Random(1)[0];
      H(0, 0) = VecXf::Random(1).cwiseAbs()[0];
      grad(0) = res;
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      return res * res;
    };
    double x = 0;
    lm::Options options;
    options.max_duration_ms = 15;
    const auto &out = lm::Optimize(x, loss, options);
    SuccessChecks(out, 0, StopReason::kTimedOut);
  }
}

/// Common checks on an early failure
void FailureChecks(const auto &out, StopReason expected_stop = StopReason::kSolverFailed) {
  std::cout << out.StopReasonDescription() << "\n";
  REQUIRE(!out.Succeeded());
  REQUIRE(!out.Converged());
  REQUIRE(out.num_iters == 0);
  REQUIRE(out.errs2.empty());
  REQUIRE(out.successes.empty());
  REQUIRE(out.deltas2.empty());
  REQUIRE(out.stop_reason == expected_stop);
}

void TestFailures() {
  // NaN in grad
  {
    std::cout << "**** Testing NaNs in Jt * res\n";
    auto loss = [&](const auto &x, auto &grad, auto &H) {
      double res = x - 2;
      H(0, 0) = 1;
      grad(0) = NAN;  // a NaN? Yeah, that's bad NaN.
      return res * res;
    };
    double x = 1;
    const auto &out = lm::Optimize(x, loss);
    FailureChecks(out, StopReason::kSystemHasNaNOrInf);
  }
  // Infinity in grad
  {
    std::cout << "**** Testing Infinity in grad\n";
    auto loss = [&](const auto &x, auto &grad, auto &H) {
      double res = x - 2;
      H(0, 0) = 1;
      grad(0) = std::numeric_limits<double>::infinity();
      return res * res;
    };
    double x = 1;
    const auto &out = lm::Optimize(x, loss);
    FailureChecks(out, StopReason::kSystemHasNaNOrInf);
  }
  // Infinity in grad
  {
    std::cout << "**** Testing Infinity in res\n";
    auto loss = [&](const auto &x, auto &grad, auto &H) {
      double res = x + std::numeric_limits<double>::infinity();
      H(0, 0) = 1;
      grad(0) = std::numeric_limits<double>::infinity();
      return res * res;
    };
    double x = 1;
    const auto &out = lm::Optimize(x, loss);
    FailureChecks(out, StopReason::kSystemHasNaNOrInf);
  }
  // Infinity in res*res
  {
    std::cout << "**** Testing Infinity in res\n";
    auto loss = [&](const auto &x, auto &grad, auto &H) {
      double res = x + 1;
      H(0, 0) = 1;
      grad(0) = res;
      return std::numeric_limits<double>::infinity();
    };
    double x = 1;
    const auto &out = lm::Optimize(x, loss);
    FailureChecks(out, StopReason::kSystemHasNaNOrInf);
  }
  // Forgot to update H
  /*{
    std::cout << "**** Testing Forgot to update H\n";
    auto loss = [&](const auto &x, auto &, auto &) {
      double res = x - 2;
      // Let's forget to set H
      return res * res;
    };
    double x = 1;
    const auto &out = lm::Optimize(x, loss);
    FailureChecks(out, StopReason::kSkipped);
  }
  // Non-invertible H
  {
    std::cout << "**** Testing Non-invertible H\n";
    auto loss = [&](const auto &x, auto &grad, auto &H) {
      Vec2 res(x[0] - 2, -x[1] + 1);
      H = Mat2::Identity();
      H(1, 1) = -1;
      grad = res;
      return res.squaredNorm();
    };
    Vec2 x(1, 1);
    const auto &out = lm::Optimize(x, loss);
    FailureChecks(out, StopReason::kMaxConsecFails);
  }*/
  // No residuals
  {
    std::cout << "**** No residuals\n";
    auto loss = [&](const auto &, auto &, auto &) {
      return VecX();  // no residuals
    };
    double x = 1;
    const auto &out = lm::Optimize(x, loss);
    FailureChecks(out, StopReason::kSkipped);
  }
  // Empty x
  {
    std::cout << "**** Testing Empty x\n";
    auto loss = [&](const auto &x, auto &grad, auto &H) {
      double res = x[0] - 2;
      H(0, 0) = 1;
      grad(0) = res;
      return res * res;
    };
    std::vector<float> empty;
    const auto &out = lm::Optimize(empty, loss);
    FailureChecks(out, StopReason::kSkipped);
  }
// Out of memory (only on linux, not sure why it crashes on MacOS..)
#if (defined(LINUX) || defined(__linux__))
  {
    std::cout << "**** Testing Out of Memory x\n";
    auto loss = [&](const auto &x, auto &grad, auto &H) {
      double res = x[0] - 2;
      H(0, 0) = 1;
      grad(0) = res;
      return res * res;
    };
    std::vector<double> too_large;
    try {
      // unless you're Elon and can afford that memoryfor a dense H matrix
      too_large.resize(100000);
      const auto &out = lm::Optimize(too_large, loss);
      FailureChecks(out, StopReason::kOutOfMemory);
    } catch (const std::bad_alloc &e) {
      std::cout << "CAN'T EVEN ALLOCATE x...\n";
    }
  }
#endif
}

TEST_CASE("tinyopt_basic_success") { TestSuccess(); }

TEST_CASE("tinyopt_basic_failures") { TestFailures(); }