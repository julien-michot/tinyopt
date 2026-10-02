// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <type_traits>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <tinyopt/params_wrapper.h>
#include <tinyopt/tinyopt.h>

using Catch::Approx;
using namespace tinyopt;

template <typename T>
struct Pose2 {
  using Scalar = T;
  using Vec2 = Vector<T, 2>;
  using Tangent = Vector<T, 3>;
  static constexpr Index Dims = 3;

  Pose2() : translation(Vec2::Zero()), orientation(Vec2(1, 0)) {}

  template <typename T2>
  static auto cast(const Pose2& pose) {
    Pose2<T2> result;
    result.translation = pose.translation.template cast<T2>();
    result.orientation = pose.orientation.template cast<T2>();
    return result;
  }

  Pose2& operator+=(const Tangent& delta) {
    translation += delta.template head<2>();

    const T tangent = delta[2];
    const T denominator = T(1) + tangent * tangent;
    const T cosine = (T(1) - tangent * tangent) / denominator;
    const T sine = T(2) * tangent / denominator;
    const Vec2 previous_orientation = orientation;
    orientation[0] = cosine * previous_orientation[0] - sine * previous_orientation[1];
    orientation[1] = sine * previous_orientation[0] + cosine * previous_orientation[1];
    return *this;
  }

  Vec2 translation;
  Vec2 orientation;
};

TEST_CASE("tinyopt_params_wrapper_updates_external_manifold_pose") {
  Pose2<double> pose;
  ParamsWrapper<Pose2<double>&> parameters(pose);

  static_assert(traits::params_trait<decltype(parameters)>::Dims == Dynamic);
  REQUIRE(parameters.dims() == 3);
  static_assert(std::is_same_v<decltype(parameters.params), Pose2<double>&>);

  auto residuals = [](const auto& x) {
    using T = typename std::decay_t<decltype(x.params)>::Scalar;
    Vector<T, 4> result;
    result.template head<2>() = x.params.translation - Vector<T, 2>(2, -1);
    result.template tail<2>() = x.params.orientation - Vector<T, 2>(0.8, 0.6);
    return result;
  };

  Options options;
  options.lm.damping_init = 1e-3;
  const auto& out = Optimize(parameters, residuals, options);

  REQUIRE(out.Succeeded());
  REQUIRE(out.Converged());
  REQUIRE((pose.translation - Vector<double, 2>(2, -1)).norm() == Approx(0.0).margin(1e-5));
  REQUIRE((pose.orientation - Vector<double, 2>(0.8, 0.6)).norm() == Approx(0.0).margin(1e-5));
}