// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <string>

#include <Eigen/Core>

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <tinyopt/3rdparty/traits/sophus.h>
#include <tinyopt/losses/robust_norms.h>
#include <tinyopt/tinyopt.h>

#include "iterations.h"
#include "options.h"
#include "robust_pose.h"

using namespace tinyopt;
using namespace tinyopt::benchmark;
using namespace tinyopt::benchmark::robust_pose;

namespace {

struct PoseLoss {
  const std::vector<Observation>& observations;
  bool robust;
  double delta;

  Cost operator()(const Pose& pose, auto& gradient, auto& hessian) const {
    double cost = 0;
    if constexpr (!traits::is_nullptr_v<decltype(gradient)>) gradient.setZero();
    if constexpr (!traits::is_nullptr_v<decltype(hessian)>) hessian.setZero();

    const Eigen::Matrix3d rotation = pose.so3().matrix();
    for (const Observation& observation : observations) {
      const Eigen::Vector3d residual = pose * observation.world_point - observation.measurement;
      const double squared_norm = residual.squaredNorm();
      double scale = 1;
      double scale_derivative = 0;
      if (robust && squared_norm > delta * delta) {
        const double inverse_norm = 1.0 / std::sqrt(squared_norm);
        scale = std::sqrt(2.0 * delta * inverse_norm - delta * delta * inverse_norm * inverse_norm);
        scale_derivative =
            -delta * inverse_norm * inverse_norm * inverse_norm +
            delta * delta * inverse_norm * inverse_norm * inverse_norm * inverse_norm;
        cost += 0.5 * scale * scale * squared_norm;
      } else {
        cost += 0.5 * squared_norm;
      }

      if constexpr (!traits::is_nullptr_v<decltype(gradient)>) {
        Eigen::Matrix3d skew;
        skew << 0, -observation.world_point[2], observation.world_point[1],
            observation.world_point[2], 0, -observation.world_point[0],
            -observation.world_point[1], observation.world_point[0], 0;
        Eigen::Matrix<double, 3, 6> pose_jacobian;
        pose_jacobian.leftCols<3>() = rotation;
        pose_jacobian.rightCols<3>().noalias() = -rotation * skew;
        Eigen::Matrix3d residual_jacobian = scale * Eigen::Matrix3d::Identity();
        if (scale_derivative != 0)
          residual_jacobian.noalias() +=
              (scale_derivative / scale) * residual * residual.transpose();
        const Eigen::Matrix<double, 3, 6> jacobian = residual_jacobian * pose_jacobian;
        const Eigen::Vector3d scaled_residual = scale * residual;
        if constexpr (!traits::is_nullptr_v<decltype(gradient)>)
          gradient.noalias() += jacobian.transpose() * scaled_residual;
        if constexpr (!traits::is_nullptr_v<decltype(hessian)>)
          hessian.noalias() += jacobian.transpose() * jacobian;
      }
    }
    return Cost(cost, static_cast<int>(3 * observations.size()));
  }
};

}  // namespace

TEMPLATE_TEST_CASE("PoseOptimization", "[benchmark][robust][pose]", Vec6) {
  const Pose ground_truth = GroundTruth();
  const int observation_count = GENERATE(20, 50, 100);
  const int outliers = observation_count / 4;
  const auto observations =
      MakeObservations(ground_truth, observation_count - outliers, outliers, 0.01, 50.0);
  const double delta = 0.3;

  Options options = CreateOptions(false);
  options.stop.max_iters = 100;
  options.stop.max_consec_failures = 20;

  for (const bool robust : {false, true}) {
    const PoseLoss loss{observations, robust, delta};
    Pose verification_pose = InitialPose(ground_truth);
    lm::Optimizer<Mat6> verification_optimizer(options);
    const auto scalar_loss = [&loss](const Pose& pose, auto& gradient) {
      std::nullptr_t null_hessian{};
      return loss(pose, gradient, null_hessian).cost;
    };
    REQUIRE(
        diff::CheckGradient(verification_pose, scalar_loss, 1e-4, diff::Method::kCentral, false));
    const auto verification = verification_optimizer(verification_pose, loss);
    REQUIRE(verification.Succeeded());
    REQUIRE(verification.Converged());
    const std::string loss_name = robust ? "Huber" : "L2";
    PrintIterations("Robust pose", std::to_string(observation_count) + "o " + loss_name,
                    "tinyopt", verification.num_iters, verification.Converged());

    lm::Optimizer<Mat6> optimizer(options);
    BENCHMARK(std::to_string(observation_count) + " obs " + loss_name) {
      Pose pose = InitialPose(ground_truth);
      optimizer.reset();
      const auto result = optimizer(pose, loss);
      return result.final_cost.cost;
    };
  }
}
