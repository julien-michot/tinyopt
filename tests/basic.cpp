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

#if CATCH2_VERSION == 2
#include <catch2/catch.hpp>
#else
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#endif

#include "tinyopt/tinyopt.h"

using namespace tinyopt;

/// Common checks on an successful optimization
void SuccessChecks(const auto &out, int min_num_iters = 1,
                   StopReason expected_stop = StopReason::kMinGradNorm) {
  REQUIRE(out.last_err2 < 1e-5);
  REQUIRE(out.Succeeded());
  REQUIRE(out.Converged());
  REQUIRE(out.stop_reason == StopReason::kMinGradNorm);
  REQUIRE(out.num_iters >= min_num_iters);
  REQUIRE(out.last_JtJ(0, 0) > 0);  // was exported
  REQUIRE(out.errs2.size() == size_t(out.num_iters + 1));
  REQUIRE(out.successes.size() == size_t(out.num_iters + 1));
  REQUIRE(out.deltas2.size() == size_t(out.num_iters + 1));
  std::cout << out.StopReasonDescription() << "\n";
  REQUIRE(out.stop_reason == expected_stop);
}

void TestSuccess() {
  // Normal case using LM
  {
    std::cout << "**** Normal Test Case LM \n";
    auto loss = [&](const auto &x, auto &JtJ, auto &Jt_res) {
      double res = x - 2;
      JtJ(0, 0) = 1;
      Jt_res(0) = res;
      return res * res;
    };
    double x = 1;
    const auto &out = Optimize(x, loss);
    SuccessChecks(out);
  }
  // Normal case using LM
  {
    std::cout << "**** Normal Test Case GN\n";
    auto loss = [&](const auto &x, auto &JtJ, auto &Jt_res) {
      double res = x - 2;
      JtJ(0, 0) = 1;
      Jt_res(0) = res;
      return res * res;
    };
    double x = 1;
    const auto &out = gn::Optimize(x, loss);
    SuccessChecks(out);
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
  // NaN in Jt_res
  {
    std::cout << "**** Testing NaNs in Jt * res\n";
    auto loss = [&](const auto &x, auto &JtJ, auto &Jt_res) {
      double res = x - 2;
      JtJ(0, 0) = 1;
      Jt_res(0) = NAN;  // a NaN? Yeah, that's bad NaN.
      return res * res;
    };
    double x = 1;
    const auto &out = Optimize(x, loss);
    FailureChecks(out, StopReason::kSystemHasNaNOrInf);
  }
  // Infinity in Jt_res
  {
    std::cout << "**** Testing Infinity in Jt_res\n";
    auto loss = [&](const auto &x, auto &JtJ, auto &Jt_res) {
      double res = x - 2;
      JtJ(0, 0) = 1;
      Jt_res(0) = std::numeric_limits<double>::infinity();
      return res * res;
    };
    double x = 1;
    const auto &out = Optimize(x, loss);
    FailureChecks(out, StopReason::kSystemHasNaNOrInf);
  }
  // Infinity in Jt_res
  {
    std::cout << "**** Testing Infinity in res\n";
    auto loss = [&](const auto &x, auto &JtJ, auto &Jt_res) {
      double res = x + std::numeric_limits<double>::infinity();
      JtJ(0, 0) = 1;
      Jt_res(0) = std::numeric_limits<double>::infinity();
      return res * res;
    };
    double x = 1;
    const auto &out = Optimize(x, loss);
    FailureChecks(out, StopReason::kSystemHasNaNOrInf);
  }
  // Infinity in res*res
  {
    std::cout << "**** Testing Infinity in res\n";
    auto loss = [&](const auto &x, auto &JtJ, auto &Jt_res) {
      double res = x + 1;
      JtJ(0, 0) = 1;
      Jt_res(0) = res;
      return std::numeric_limits<double>::infinity();
    };
    double x = 1;
    const auto &out = Optimize(x, loss);
    FailureChecks(out, StopReason::kSystemHasNaNOrInf);
  }
  // Forgot to update JtJ
  /*{
    std::cout << "**** Testing Forgot to update JtJ\n";
    auto loss = [&](const auto &x, auto &, auto &) {
      double res = x - 2;
      // Let's forget to set JtJ
      return res * res;
    };
    double x = 1;
    const auto &out = Optimize(x, loss);
    FailureChecks(out, StopReason::kSkipped);
  }
  // Non-invertible JtJ
  {
    std::cout << "**** Testing Non-invertible JtJ\n";
    auto loss = [&](const auto &x, auto &JtJ, auto &Jt_res) {
      Vec2 res(x[0] - 2, -x[1] + 1);
      JtJ = Mat2::Identity();
      JtJ(1, 1) = -1;
      Jt_res = res;
      return res.squaredNorm();
    };
    Vec2 x(1, 1);
    const auto &out = Optimize(x, loss);
    FailureChecks(out, StopReason::kMaxConsecFails);
  }*/
  // No residuals
  {
    std::cout << "**** No residuals\n";
    auto loss = [&](const auto &, auto &, auto &) {
      return VecX();  // no residuals
    };
    double x = 1;
    const auto &out = Optimize(x, loss);
    FailureChecks(out, StopReason::kSkipped);
  }
  // Empty x
  {
    std::cout << "**** Testing Empty x\n";
    auto loss = [&](const auto &x, auto &JtJ, auto &Jt_res) {
      double res = x[0] - 2;
      JtJ(0, 0) = 1;
      Jt_res(0) = res;
      return res * res;
    };
    std::vector<float> empty;
    const auto &out = Optimize(empty, loss);
    FailureChecks(out, StopReason::kSkipped);
  }
}

TEST_CASE("tinyopt_basic_success") { TestSuccess(); }

TEST_CASE("tinyopt_basic_failures") { TestFailures(); }