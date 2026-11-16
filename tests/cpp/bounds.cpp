// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <type_traits>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <tinyopt/diff/gradient_check.h>

#include "bounds_types.h"

using Catch::Approx;
using namespace tinyopt;
using namespace tinyopt_bounds_test;

TEST_CASE("Static member bounds constrain manifold values", "[bounds][member]") {
  AnglePose<double> pose;
  auto objective = [](const auto& value, auto& gradient, auto& hessian) {
    using Scalar = std::decay_t<decltype(value.angle)>;
    const Scalar residual = value.angle - Scalar(2);
    if constexpr (!traits::is_nullptr_v<decltype(gradient)>) {
      gradient[0] = Scalar(2) * residual;
    }
    if constexpr (!traits::is_nullptr_v<decltype(hessian)>) {
      hessian(0, 0) = Scalar(2);
    }
    return residual * residual;
  };
  REQUIRE(diff::CheckGradient(pose, objective));

  Options options;
  options.stop.max_iters = 10;
  lm::Optimizer<Matrix<double, 1, 1>> optimizer(options);
  const auto summary = optimizer.Optimize(pose, objective);

  REQUIRE(summary.Succeeded());
  REQUIRE(summary.Converged());
  REQUIRE(pose.angle == Approx(0.5).margin(1e-8));
  const auto [last_lower, last_upper] = optimizer.lastBounds();
  REQUIRE(last_lower[0] == Approx(-1.0).margin(1e-8));
  REQUIRE(last_upper[0] == Approx(0.0).margin(1e-8));

  AnglePose<double> overridden_pose;
  optimizer.setBounds(-0.2, 0.2);
  const auto overridden_summary = optimizer.Optimize(overridden_pose, objective);
  REQUIRE(overridden_summary.Succeeded());
  REQUIRE(overridden_pose.angle == Approx(2.0).margin(1e-8));

  optimizer.clearBounds();
  optimizer.reset();
  AnglePose<double> restored_pose;
  const auto restored_summary = optimizer.Optimize(restored_pose, objective);
  REQUIRE(restored_summary.Succeeded());
  REQUIRE(restored_pose.angle == Approx(0.5).margin(1e-8));
}

TEST_CASE("Scalar member bounds support fixed-size float systems", "[bounds][member]") {
  BoundedScalar<float> state;
  auto loss = [](const auto& value, auto& gradient, auto& hessian) {
    const auto residual = value.value - 1.0f;
    if constexpr (!traits::is_nullptr_v<decltype(gradient)>) gradient[0] = 2.0f * residual;
    if constexpr (!traits::is_nullptr_v<decltype(hessian)>) hessian(0, 0) = 2.0f;
    return residual * residual;
  };

  Options options(Options::Solver::GaussNewton);
  options.stop.max_iters = 1;
  gn::Optimizer<Matrix<float, 1, 1>> optimizer(options);
  const auto summary = optimizer.Optimize(state, loss);

  REQUIRE(summary.Succeeded());
  REQUIRE(state.value == Approx(0.5f).margin(1e-5f));
  const auto [lower, upper] = optimizer.lastBounds();
  REQUIRE(lower[0] == Approx(-0.25f));
  REQUIRE(upper[0] == Approx(0.25f));
}

TEST_CASE("Dynamic member bounds constrain dense optimizer steps", "[bounds][member]") {
  DynamicBoundedState state(3);
  const Vec3 target(2.0, -1.0, 3.0);
  Mat3 information = Mat3::Identity();
  information(0, 0) = 2.0;
  information(0, 1) = 0.5;
  information(1, 0) = 0.5;
  information(1, 1) = 1.5;
  auto objective = [&](const auto& value, auto& gradient) {
    using Scalar = typename std::decay_t<decltype(value.values)>::Scalar;
    const auto difference = value.values - target.template cast<Scalar>();
    const auto typed_information = information.template cast<Scalar>();
    if constexpr (!traits::is_nullptr_v<decltype(gradient)>) {
      gradient.noalias() = typed_information * difference;
    }
    return 0.5 * difference.dot(typed_information * difference);
  };
  REQUIRE(diff::CheckGradient(state, objective));

  auto loss = [&](const DynamicBoundedState<double>& value, VecX& gradient, MatX& hessian) {
    const Vec3 difference = value.values - target;
    gradient = information * difference;
    hessian = information;
    return Cost(0.5 * difference.dot(information * difference), 3);
  };

  Options options(Options::Solver::GaussNewton);
  options.opt.active_set_ratio = 0.2f;
  options.stop.max_iters = 1;
  gn::Optimizer<MatX> optimizer(options);
  const auto summary = optimizer(state, loss);

  REQUIRE(summary.Succeeded());
  REQUIRE(state.values[0] == Approx(1.0).margin(1e-8));
  REQUIRE(state.values[1] == Approx(-2.0 / 3.0).margin(1e-8));
  REQUIRE(state.values[2] == Approx(3.0).margin(1e-8));
}
