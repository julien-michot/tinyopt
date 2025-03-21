// Copyright (C) 2025 Julien Michot. All Rights reserved.

#include <cmath>
#include <utility>


#if CATCH2_VERSION == 2
#include <catch2/catch.hpp>
#else
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#endif

#include "tinyopt/tinyopt.h"

using namespace tinyopt::lm;

using Catch::Approx;
using Vec2f = Eigen::Vector<float, 2>;


std::vector<Vec2f> CreateCirle(int n, float r, const Vec2f &center = Vec2f::Zero(), float noise = 0)
{
  std::vector<Vec2f> obs(n);
  float angle = 0;
  for (auto &o : obs) {
    o = center + r * Vec2f(cosf(angle), sinf(angle)) + noise * Vec2f::Random();
    angle += 2 * M_PI / (n - 1);
  }
  return obs;
}

void TestFitCircle() {
  const float radius = 2;
  const Vec2f center(2, 7);
  const auto obs = CreateCirle(radius, 10, center, 1e-5);

  // Loss is
  auto loss = [&obs](const auto &x) {
    using T = std::remove_const_t<std::remove_reference_t<decltype(x[0])>>; // recover Jet type
    const auto &center = x.template head<2>();
    const auto radius2 = x.z() * x.z();
    Eigen::Vector<T, Eigen::Dynamic> residuals(obs.size());
    for (size_t i = 0; i < obs.size(); ++i) {
      residuals(i) = (obs[i].cast<T>() - center).squaredNorm() - radius2;
    }
    return residuals;
  };

  Eigen::Vector<double, 3> x(1, 6, 1); // center (x, y) + radius
  const auto &out = AutoLM(x, loss);

  REQUIRE(out.Succeeded());
  REQUIRE(x.x() == Approx(center.x()).epsilon(1e-5));
  REQUIRE(x.y() == Approx(center.y()).epsilon(1e-5));
  REQUIRE(x.z() == Approx(radius).epsilon(1e-5));
}


TEST_CASE("tinyopt_fitcircle") {
  TestFitCircle();
}