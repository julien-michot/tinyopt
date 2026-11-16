// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <limits>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <tinyopt/diff/gradient_check.h>
#include <tinyopt/tinyopt.h>

using Catch::Approx;
using namespace tinyopt;

TEST_CASE("Runtime bounds project static first-order steps", "[bounds][runtime]") {
  Vec2 x = Vec2::Zero();
  const Vec2 target(2.0, 1.0);
  auto objective = [&](const auto& value, auto& gradient) {
    const Vec2 difference = value - target;
    if constexpr (!traits::is_nullptr_v<decltype(gradient)>) gradient = difference;
    return 0.5 * difference.squaredNorm();
  };
  REQUIRE(diff::CheckGradient(x, objective));

  gd::Optimizer<Vec2>::Options options;
  options.gd.lr = 2.0;
  options.line_search.method = Options::LineSearchMethod::None;
  options.stop.max_iters = 1;
  double max_observed_step = 0.0;
  options.stop.step_callback = [&](const auto& step, bool) {
    max_observed_step =
        std::max(max_observed_step, std::abs(static_cast<double>(step[0])));
    return false;
  };
  gd::Optimizer<Vec2> optimizer(options);
  optimizer.fix({1});
  Vec2 lower = Vec2::Constant(-std::numeric_limits<double>::infinity());
  Vec2 upper = Vec2::Constant(std::numeric_limits<double>::infinity());
  upper[0] = 0.5;
  lower[1] = 0.1;
  optimizer.setBounds(lower, upper);

  const auto summary = optimizer(x, objective);

  REQUIRE(summary.Succeeded());
  REQUIRE(x[0] == Approx(1.0).margin(1e-8));
  REQUIRE(x[1] == Approx(0.0).margin(1e-8));
  REQUIRE(max_observed_step <= 0.5);
}

TEST_CASE("Quasi-Newton and conjugate-gradient steps respect bounds", "[bounds][first-order]") {
  const Vec2 target(2.0, -1.0);
  auto objective = [&](const auto& value, auto& gradient) {
    const Vec2 difference = value - target;
    if constexpr (!traits::is_nullptr_v<decltype(gradient)>) gradient = difference;
    return 0.5 * difference.squaredNorm();
  };
  Vec2 lower = Vec2::Constant(-std::numeric_limits<double>::infinity());
  Vec2 upper = Vec2::Constant(std::numeric_limits<double>::infinity());
  lower[0] = 0.0;
  upper[0] = 0.5;

  double max_observed_step = 0.0;
  auto step_callback = [&](const auto& step, bool) {
    max_observed_step =
        std::max(max_observed_step, std::abs(static_cast<double>(step[0])));
    return false;
  };
  auto verify = [&](auto& optimizer, const char* name) {
    INFO(name);
    Vec2 x = Vec2::Zero();
    optimizer.setBounds(lower, upper);
    const auto summary = optimizer(x, objective);
    REQUIRE(summary.Succeeded());
    REQUIRE(max_observed_step <= 0.5);
    REQUIRE(x[0] <= 1.0 + 1e-8);
    max_observed_step = 0.0;
  };

  cg::Optimizer<Vec2>::Options cg_options;
  cg_options.cg.step_size = 2.0;
  cg_options.stop.max_iters = 1;
  cg_options.line_search.method = Options::LineSearchMethod::None;
  cg_options.stop.step_callback = step_callback;
  cg::Optimizer<Vec2> cg_optimizer(cg_options);
  verify(cg_optimizer, "Conjugate gradient");

  bfgs::Optimizer<Vec2>::Options bfgs_options;
  bfgs_options.bfgs.step_size = 2.0;
  bfgs_options.stop.max_iters = 1;
  bfgs_options.line_search.method = Options::LineSearchMethod::None;
  bfgs_options.stop.step_callback = step_callback;
  bfgs::Optimizer<Vec2> bfgs_optimizer(bfgs_options);
  verify(bfgs_optimizer, "BFGS");

  lbfgs::Optimizer<Vec2>::Options lbfgs_options;
  lbfgs_options.lbfgs.step_size = 2.0;
  lbfgs_options.stop.max_iters = 1;
  lbfgs_options.line_search.method = Options::LineSearchMethod::None;
  lbfgs_options.stop.step_callback = step_callback;
  lbfgs::Optimizer<Vec2> lbfgs_optimizer(lbfgs_options);
  verify(lbfgs_optimizer, "L-BFGS");
}
