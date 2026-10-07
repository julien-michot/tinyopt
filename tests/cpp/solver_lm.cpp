// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#if CATCH2_VERSION == 2
#include <catch2/catch.hpp>
#else
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#endif

#include <tinyopt/diff/num_diff.h>
#include <tinyopt/optimizers/lm.h>

using Catch::Approx;
using namespace tinyopt;

TEST_CASE("tinyopt_solver_lm_numdiff") {
  lm::Optimizer<Mat2> optimizer;
  using Vec = lm::Optimizer<Mat2>::Grad_t;
  Vec x = Vec::Zero();
  const Vec2 target(4, 5);
  const auto residuals = [&](const auto &value) { return (value - target).eval(); };

  REQUIRE(optimizer.Build(x, diff::CreateNumDiffFunc2(x, residuals)));
  const auto maybe_dx = optimizer.SolveLM();
  REQUIRE(maybe_dx.has_value());
  REQUIRE(maybe_dx->x() == Approx(target.x()).margin(1e-2));
  REQUIRE(maybe_dx->y() == Approx(target.y()).margin(1e-2));
}

TEST_CASE("tinyopt_solver_lm_skip_rebuild") {
  lm::Optimizer<Mat2> optimizer;
  const Vec2 x = Vec2::Zero();
  const Vec2 target(4, 5);
  int gradient_updates = 0;
  const auto loss = [&](const auto &value, auto &gradient, auto &hessian) {
    const auto residual = (value - target).eval();
    if constexpr (!traits::is_nullptr_v<decltype(gradient)>) {
      gradient = residual;
      hessian = Mat2::Identity();
      ++gradient_updates;
    }
    return residual;
  };

  REQUIRE(optimizer.Build(x, loss));
  REQUIRE(gradient_updates == 1);
  optimizer.Rebuild(false);
  REQUIRE(optimizer.Build(x, loss));
  REQUIRE(gradient_updates == 1);
  const auto maybe_dx = optimizer.SolveLM();
  REQUIRE(maybe_dx.has_value());
  REQUIRE(maybe_dx->x() == Approx(target.x()).margin(1e-2));
  REQUIRE(maybe_dx->y() == Approx(target.y()).margin(1e-2));
}

TEST_CASE("tinyopt_solver_lm_increases_damping_on_bad_step") {
  lm::Optimizer<Mat2> optimizer;
  const Vec2 x = Vec2::Zero();
  const Vec2 target(4.0, 5.0);
  const auto residuals = [&](const auto &value, auto &gradient, auto &hessian) {
    const auto residual = (value - target).eval();
    if constexpr (!traits::is_nullptr_v<decltype(gradient)>) {
      gradient = residual;
      hessian = Mat2::Identity();
    }
    return residual;
  };

  REQUIRE(optimizer.Build(x, residuals));
  const auto before = optimizer.stateAsString();
  optimizer.BadStep();
  REQUIRE(optimizer.stateAsString() != before);
}