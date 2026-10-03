// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <array>

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <tinyopt/tinyopt.h>

#include "bundle_adjustment.h"
#include "options.h"

using namespace tinyopt;
using namespace tinyopt::benchmark;
using namespace tinyopt::benchmark::bundle_adjustment;

namespace tinyopt::traits {

template <>
struct params_trait<tinyopt::benchmark::bundle_adjustment::Problem> {
  using Scalar = double;
  static constexpr Index Dims = Dynamic;

  static Index dims(const tinyopt::benchmark::bundle_adjustment::Problem&) {
    return 6 * (tinyopt::benchmark::bundle_adjustment::CameraCount - 1) +
           3 * (tinyopt::benchmark::bundle_adjustment::PointCount - 1);
  }

  static void PlusEq(tinyopt::benchmark::bundle_adjustment::Problem& problem, const auto& delta) {
    constexpr int CameraOffset = 6 * (tinyopt::benchmark::bundle_adjustment::CameraCount - 1);
    for (int camera = 1; camera < tinyopt::benchmark::bundle_adjustment::CameraCount; ++camera)
      problem.initial_cameras[camera] += delta.template segment<6>((camera - 1) * 6);
    for (int point = 1; point < tinyopt::benchmark::bundle_adjustment::PointCount; ++point)
      problem.initial_points[point] += delta.template segment<3>(CameraOffset + (point - 1) * 3);
  }
};

}  // namespace tinyopt::traits

namespace {

using LocalParameters = Eigen::Matrix<double, 9, 1>;
using LocalJacobian = Eigen::Matrix<double, 2, 9>;
using LocalJet = diff::Jet<double, 9>;

LocalJacobian ProjectionJacobian(const LocalParameters& parameters, const ImagePoint& measurement) {
  Eigen::Matrix<LocalJet, 9, 1> differentiated;
  for (int index = 0; index < 9; ++index) {
    differentiated[index] = LocalJet(parameters[index]);
    differentiated[index].v.setZero();
    differentiated[index].v[index] = 1.0;
  }

  const Eigen::Matrix<LocalJet, 6, 1> camera = differentiated.template head<6>();
  const Eigen::Matrix<LocalJet, 3, 1> point = differentiated.template tail<3>();
  const Eigen::Matrix<LocalJet, 2, 1> residual =
      Project(camera, point) - measurement.cast<LocalJet>();
  LocalJacobian jacobian;
  for (int row = 0; row < 2; ++row) jacobian.row(row) = residual[row].v.transpose();
  return jacobian;
}

struct Loss {
  Cost operator()(const Problem& problem, auto& gradient, SparseMat& hessian) const {
    constexpr int CameraOffset = 6 * (CameraCount - 1);
    double squared_error = 0;

    if constexpr (!traits::is_nullptr_v<decltype(gradient)>) {
      gradient.setZero();
      hessian.setZero();
    }

    // Simple implementation of the cost function and its derivatives for bundle adjustment.
    for (const auto& observation : problem.observations) {
      const Camera& camera = problem.initial_cameras[observation.camera];
      const Point& point = problem.initial_points[observation.point];
      const ImagePoint residual = Project(camera, point) - observation.measurement;
      squared_error += residual.squaredNorm();

      if constexpr (!traits::is_nullptr_v<decltype(gradient)>) {
        LocalParameters local;
        local << camera, point;
        const LocalJacobian jacobian = ProjectionJacobian(local, observation.measurement);
        std::array<Index, 9> global_indices;
        global_indices.fill(-1);
        if (observation.camera != 0) {
          for (int index = 0; index < 6; ++index)
            global_indices[index] = (observation.camera - 1) * 6 + index;
        }
        if (observation.point != 0) {
          for (int index = 0; index < 3; ++index)
            global_indices[6 + index] = CameraOffset + (observation.point - 1) * 3 + index;
        }

        for (int row = 0; row < 9; ++row) {
          const Index global_row = global_indices[row];
          if (global_row < 0) continue;
          gradient[global_row] += jacobian.col(row).dot(residual);
          for (int column = 0; column < 9; ++column) {
            const Index global_column = global_indices[column];
            if (global_column >= 0)
              hessian.coeffRef(global_row, global_column) +=
                  jacobian.col(row).dot(jacobian.col(column));
          }
        }
      }
    }
    return Cost(0.5 * squared_error, 2 * ObservationCount);
  }
};

}  // namespace

TEST_CASE("BA", "[benchmark][bundle-adjustment][sparse]") {
  const Problem initial_problem = MakeProblem();
  Problem verification_problem = initial_problem;
  const Loss loss;
  Options options = CreateOptions();
  options.stop.max_iters = 10;
  options.lm.jacobi_scaling = true;
  const double reference_initial_cost = ReferenceReprojectionCost(initial_problem);

  const Observation& check_observation = initial_problem.observations[PointCount + 1];
  auto local_cost = [&check_observation](const LocalParameters& local, auto& gradient) {
    Camera camera = local.head<6>();
    Point point = local.tail<3>();
    const ImagePoint residual = Project(camera, point) - check_observation.measurement;
    if constexpr (!traits::is_nullptr_v<decltype(gradient)>) {
      const LocalJacobian jacobian = ProjectionJacobian(local, check_observation.measurement);
      gradient = jacobian.transpose() * residual;
    }
    return 0.5 * residual.squaredNorm();
  };
  LocalParameters check_parameters;
  check_parameters << initial_problem.initial_cameras[1], initial_problem.initial_points[1];
  REQUIRE(diff::CheckGradient(check_parameters, local_cost, 1e-4, diff::Method::kCentral, false));

  lm::Optimizer<SparseMat> verification_optimizer(options);
  const auto& verification = verification_optimizer(verification_problem, loss);
  std::nullptr_t null_gradient{};
  SparseMat unused_hessian;
  const double initial_cost = loss(initial_problem, null_gradient, unused_hessian).cost;
  REQUIRE(verification.Succeeded());
  REQUIRE(verification.Converged());
  REQUIRE(initial_cost == Catch::Approx(reference_initial_cost).margin(1e-8));
  REQUIRE(verification.final_cost.cost < initial_cost * 1e-4);
  REQUIRE(verification.final_cost.cost < 1e-6);

  BENCHMARK("5 cams, 50 pts") {
    Problem problem = initial_problem;
    lm::Optimizer<SparseMat> optimizer(options);
    const auto& result = optimizer(problem, loss);
    return result.final_cost.cost;
  };
}