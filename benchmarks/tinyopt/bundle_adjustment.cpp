// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <string>
#include <utility>

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <tinyopt/tinyopt.h>

#include "bundle_adjustment.h"
#include "bundle_adjustment_tinyopt.h"
#include "options.h"

using namespace tinyopt;
using namespace tinyopt::benchmark;
using namespace tinyopt::benchmark::bundle_adjustment;

TEST_CASE("BA", "[benchmark][bundle-adjustment][sparse]") {
  const auto dimensions = GENERATE(std::pair{5, 50}, std::pair{20, 200}, std::pair{50, 500});
  const Problem initial_problem = MakeProblem(dimensions.first, dimensions.second);
  Problem verification_problem = initial_problem;
  const TinyoptLoss loss;
  Options options = CreateOptions();
  options.stop.max_iters = 100;
  options.lm.jacobi_scaling = true;
  const double reference_initial_cost = ReferenceReprojectionCost(initial_problem);

  const int camera_window = std::min(initial_problem.CameraCount() - 1, PointCameraWindow);
  const Observation& check_observation = initial_problem.observations[camera_window];
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
  REQUIRE(verification.final_cost.cost < 1e-5);
  tinyopt::benchmark::PrintIterations("Bundle adjustment", ProblemLabel(initial_problem), "tinyopt",
                                      verification.num_iters, verification.Converged());

  const std::string label = ProblemLabel(initial_problem);
  BENCHMARK(std::string(label)) {
    Problem problem = initial_problem;
    lm::Optimizer<SparseMat> optimizer(options);
    const auto& result = optimizer(problem, loss);
    return result.final_cost.cost;
  };
}