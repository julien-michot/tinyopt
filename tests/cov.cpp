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

#include <tinyopt/tinyopt.h>

using Catch::Approx;
using namespace tinyopt;
using namespace tinyopt::nlls;

void TestCov() {
  // Testing with iso weights and manual gradient/hessian update
  {
    const Vec2 y = 10 * Vec2::Random();       // prior
    const Vec2 stdevs = Vec2::Constant(4.2);  // prior standard deviations

    auto loss = [&](const auto &x, auto &grad, auto &H) {
      Mat2 J = Mat2::Identity();
      const Vec2 res = loss::MahDiag(x - y, stdevs, &J);
      grad = J * res;
      H.diagonal() = stdevs.cwiseInverse().cwiseAbs2();  // or Jt*J
      return std::sqrt(res.dot(res));                    // return √(res.t()*res)
    };

    Vec2 x(0, 0);
    Options options;
    options.log.print_J_jet = true;
    const auto &out = lm::Optimize(x, loss, options);
    REQUIRE(out.Succeeded());
    REQUIRE(out.Converged());
    REQUIRE(out.Covariance().has_value());
    const Mat2 C = out.Covariance().value();
    REQUIRE((C.diagonal().cwiseSqrt() - stdevs).cwiseAbs().maxCoeff() == Approx(0.0).margin(1e-5));
  }
  // Testing with iso weights and AutoDiff
  {
    const Vec2 y = 10 * Vec2::Random();       // prior
    const Vec2 stdevs = Vec2::Constant(4.2);  // prior standard deviations

    auto loss = [&](const auto &x) {
      // Final error will be e = res.T * stdevs.squared().inv() * res
      return loss::MahDiag(x - y, stdevs);
    };

    Vec2 x(0, 0);
    Options options;
    options.log.print_J_jet = true;
    const auto &out = lm::Optimize(x, loss, options);
    REQUIRE(out.Succeeded());
    REQUIRE(out.Converged());
    REQUIRE(out.Covariance().has_value());
    const Mat2 C = out.Covariance().value();
    REQUIRE((C.diagonal().cwiseSqrt() - stdevs).cwiseAbs().maxCoeff() == Approx(0.0).margin(1e-5));
  }
  // Testing with a general covariance matrix and manual gradient/hessian update
  {
    const Vec2 y = 2 * Vec2::Random();  // prior
    Mat2 Cy;                            // prior covariance
    Cy << 10, 2, 2, 4;

    auto loss = [&](const auto &x, auto &grad, auto &H) {
      Mat2 J = Mat2::Identity();
      const Vec2 res = loss::Mah(x - y, Cy, &J);
      grad = J * res;                  // J is stdevs.cwiseInverse().asDiagonal()
      H = J.transpose() * J;           // Jt*J
      return std::sqrt(res.dot(res));  // return √(res.t()*res)
    };

    /*const Mat2 Lt = Cy.inverse().llt().matrixU();  // Lt
    auto loss = [&](const auto &x, auto &grad, auto &H) {
      Mat2 J = Lt;
      const Vec2 res = Lt * (x - y);
      grad = J * res;                         // J is Lt
      H = J.transpose() * J;                  // Jt*J
      return std::sqrt(res.dot(res)); // return √(res.t()*res)
    };*/

    Vec2 x(0, 0);
    Options options;
    options.log.print_J_jet = true;
    const auto &out = lm::Optimize(x, loss, options);
    REQUIRE(out.Covariance().has_value());
    const Mat2 C = out.Covariance().value();
    REQUIRE((C - Cy).cwiseAbs().maxCoeff() == Approx(0.0).margin(1e-5));
    REQUIRE(out.Succeeded());
    REQUIRE(out.Converged());
  }

  // Testing with a general covariance matrix and manual gradient/hessian update
  {
    const Vec2 y = 2 * Vec2::Random();  // prior
    Mat2 Cy;                            // prior covariance
    Cy << 10, 2, 2, 4;

    const Mat2 Lt = Cy.inverse().llt().matrixU();  // I = L*Lt
    auto loss = [&](const auto &x, auto &grad, auto &H) {
      Mat2 J = Mat2::Identity();
      const Vec2 res = loss::MahInfoU(x - y, Lt, &J);
      grad = J * res;                  // J is stdevs.cwiseInverse().asDiagonal()
      H = J.transpose() * J;           // Jt*J
      return std::sqrt(res.dot(res));  // return √(res.t()*res)
    };

    Vec2 x(0, 0);
    Options options;
    options.log.print_J_jet = true;
    const auto &out = lm::Optimize(x, loss, options);
    REQUIRE(out.Succeeded());
    REQUIRE(out.Converged());
    REQUIRE(out.Covariance().has_value());
    const Mat2 C = out.Covariance().value();
    REQUIRE((C - Cy).cwiseAbs().maxCoeff() == Approx(0.0).margin(1e-5));
  }

  // Testing with a general covariance matrix and AD
  {
    const Vec2 y = 2 * Vec2::Random();  // prior
    Mat2 Cy;                            // prior covariance
    Cy << 10, 2, 2, 4;
    const Mat2 Lt = Cy.inverse().llt().matrixU();  // Lt

    auto loss = [&](const auto &x) {
      const auto res = Lt * (x - y);  // Final error will be e = res.T * C.inv() * res
      return res.eval();              // Don't forget the .eval() since 'res' is a glue class
    };

    Vec2 x(0, 0);
    Options options;
    options.log.print_J_jet = true;
    const auto &out = lm::Optimize(x, loss, options);
    REQUIRE(out.Succeeded());
    REQUIRE(out.Converged());
    REQUIRE(out.Covariance().has_value());
    const Mat2 C = out.Covariance().value();
    REQUIRE((C - Cy).cwiseAbs().maxCoeff() == Approx(0.0).margin(1e-5));
  }
  // Testing with a general covariance matrix and AD
  {
    const Vec2 y = 2 * Vec2::Random();  // prior
    Mat2 Cy;                            // prior covariance
    Cy << 10, 2, 2, 4;

    auto loss = [&](const auto &x) {
      using T = typename std::decay_t<decltype(x)>::Scalar;
      const Matrix<T, 2, 2> C_ = Cy.template cast<T>();
      const auto res = loss::Mah(x - y, C_);  // Final error will be e = res.T * C.inv() * res
      return res.eval();  // Don't forget the .eval() since 'res' is a glue class
    };

    Vec2 x(0, 0);
    Options options;
    options.log.print_J_jet = true;
    const auto &out = lm::Optimize(x, loss, options);
    REQUIRE(out.Succeeded());
    REQUIRE(out.Converged());
    REQUIRE(out.Covariance().has_value());
    const Mat2 C = out.Covariance().value();
    REQUIRE((C - Cy).cwiseAbs().maxCoeff() == Approx(0.0).margin(1e-5));
  }
}

TEST_CASE("tinyopt_cov") { TestCov(); }