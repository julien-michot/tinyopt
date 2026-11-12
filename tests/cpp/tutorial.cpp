// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <tinyopt/params_wrapper.h>
#include <tinyopt/tinyopt.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "tutorial_common.h"

using Catch::Approx;
using namespace tinyopt;

TEST_CASE("tinyopt_tutorial_compile_examples") {
  Vec2f fixed_point = Vec2f::Zero();
  auto fixed_summary =
      tinyopt::Optimize(fixed_point, [](const auto &x) { return (x - Vec2f(1.0f, 2.0f)).eval(); });
  REQUIRE(fixed_summary.Succeeded());

  std::vector<float> dynamic_values(3, 0.0f);
  auto vector_summary =
      tinyopt::Optimize(dynamic_values, [](const auto &x) { return x[0] + x[1] + x[2] - 3.0f; });
  REQUIRE(vector_summary.Succeeded());

  std::array<Vec3, 2> landmarks{Vec3::Zero(), Vec3::Ones()};
  static_assert(traits::params_trait<decltype(landmarks)>::Dims == 6);
  traits::params_trait<decltype(landmarks)>::plus_eq(landmarks, Vector<double, 6>::Ones());

  double scalar_x = 4.0;
  auto scalar_objective = [](const auto &x) { return (x - 2.0) * (x - 2.0); };
  auto scalar_summary = tinyopt::Optimize(scalar_x, scalar_objective);
  REQUIRE(scalar_summary.Succeeded());

  Vec2 residual_x = Vec2::Zero();
  const Vec2 residual_target(3.0, -2.0);
  auto residuals = [&](const auto &value) { return (value - residual_target).eval(); };
  auto residual_summary = tinyopt::Optimize(residual_x, residuals);
  REQUIRE(residual_summary.Succeeded());

  auto loss = [&](const auto &value, auto &grad, auto &hessian) {
    const Vec2 residual = value - residual_target;
    const Mat2 J = Mat2::Identity();
    if constexpr (!traits::is_nullptr_v<decltype(grad)>) {
      grad.noalias() = J.transpose() * residual;
    }
    if constexpr (!traits::is_nullptr_v<decltype(hessian)>) {
      hessian.noalias() = J.transpose() * J;
    }
    return 0.5 * residual.squaredNorm();
  };
  auto optimizer = lm::Optimizer<Mat2>{};
  const auto manual_summary = optimizer(residual_x, loss);
  REQUIRE(manual_summary.Succeeded());

  Options svd_options;
  svd_options.linear.method = LinearSolverMethod::SVD;
  svd_options.linear.svd_relative_threshold = 1e-8;
  svd_options.exp.save_history = false;
  svd_options.exp.measure_time = false;
  REQUIRE(svd_options.linear.method == LinearSolverMethod::SVD);
  REQUIRE(svd_options.linear.svd_relative_threshold == Approx(1e-8));
  REQUIRE_FALSE(svd_options.exp.save_history);
  REQUIRE_FALSE(svd_options.exp.measure_time);

  Vec2 center(5.0, 1.0);
  Vec2 scale(2.0, 3.0);
  auto multi_summary = tinyopt::Optimize(
      center, scale, [&](const auto &c, const auto &s) { return (c - s).eval(); });
  REQUIRE(multi_summary.Succeeded());

  Landmark2D<double> landmark;
  auto custom_summary = tinyopt::Optimize(
      landmark, [](const auto &value) { return (value.position - Vec2(1.0, 2.0)).eval(); });
  REQUIRE(custom_summary.Succeeded());

  Pose2<double> pose;
  ParamsWrapper<Pose2<double> &> wrapped_pose(pose);
  auto wrapped_summary = tinyopt::Optimize(wrapped_pose, [&](const auto &p) {
    Vector<typename std::decay_t<decltype(p.params)>::Scalar, 4> residuals;
    residuals.template head<2>() = p.params.translation - Vec2(2.0, -1.0);
    residuals.template tail<2>() = p.params.orientation - Vec2(0.8, 0.6);
    return residuals;
  });
  REQUIRE(wrapped_summary.Succeeded());
  REQUIRE((pose.translation - Vec2(2.0, -1.0)).norm() == Approx(0.0).margin(1e-5));
  REQUIRE((pose.orientation - Vec2(0.8, 0.6)).norm() == Approx(0.0).margin(1e-5));
}
