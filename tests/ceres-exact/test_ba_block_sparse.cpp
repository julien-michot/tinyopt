// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <ceres/ceres.h>

#ifdef CHECK
#undef CHECK
#endif

#include <tinyopt/optimizers/lm.h>
#include <tinyopt/tinyopt.h>

#include "bundle_adjustment.h"
#include "bundle_adjustment_tinyopt.h"

using namespace tinyopt;
using namespace tinyopt::benchmark::bundle_adjustment;

namespace {

struct ReprojectionCost final : ceres::SizedCostFunction<2, 6, 3> {
  explicit ReprojectionCost(const ImagePoint& measurement) : measurement_(measurement) {}

  bool Evaluate(double const* const* parameters, double* residuals,
                double** jacobians) const override {
    const Eigen::Map<const Camera> camera(parameters[0]);
    const Eigen::Map<const Point> point(parameters[1]);
    Eigen::Matrix<double, 2, 6> camera_jacobian;
    Eigen::Matrix<double, 2, 3> point_jacobian;
    ImagePoint projected;
    ProjectionJacobians(camera, point, projected, camera_jacobian, point_jacobian);
    Eigen::Map<ImagePoint> output(residuals);
    output = projected - measurement_;
    if (jacobians != nullptr && jacobians[0] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> J(jacobians[0]);
      J = camera_jacobian;
    }
    if (jacobians != nullptr && jacobians[1] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J(jacobians[1]);
      J = point_jacobian;
    }
    return true;
  }

  ImagePoint measurement_;
};

struct CeresBAResult {
  double initial_cost;
  double final_cost;
  int iterations;
  bool converged;
  std::string message;
  std::vector<Camera> cameras;
  std::vector<Point> points;
};

inline Options MakeTinyoptLMOptions() {
  Options options(Options::Solver::LevenbergMarquardt);
  options.log.enable = false;
  options.exp.final_cov = false;
  options.exp.save_history = false;
  options.exp.measure_time = false;
  options.stop.max_iters = 100;
  options.stop.max_consec_failures = 10;
  options.stop.min_error = 0.0;
  options.stop.min_rerr_dec = 1e-6;
  options.stop.min_grad_max = 1e-9;
  options.stop.min_grad_norm2 = 0.0;
  options.stop.min_step_norm2 = 0.0;
  options.stop.min_step_rel_norm2 = 1e-16;  // (1e-8)^2 matching Ceres parameter_tolerance
  options.lm.damping_init = 1e-4;
  options.lm.jacobi_scaling = true;
  return options;
}

CeresBAResult SolveCeres(const Problem& problem, const Options& tiny_opts) {
  auto cameras = problem.initial_cameras;
  auto points = problem.initial_points;
  ceres::Problem ceres_problem;
  for (int camera = 0; camera < problem.CameraCount(); ++camera) {
    ceres_problem.AddParameterBlock(cameras[camera].data(), 6);
  }
  for (int point = 0; point < problem.PointCount(); ++point) {
    ceres_problem.AddParameterBlock(points[point].data(), 3);
  }
  ceres_problem.SetParameterBlockConstant(cameras[0].data());
  ceres_problem.SetParameterBlockConstant(points[0].data());

  auto ordering = std::make_shared<ceres::ParameterBlockOrdering>();
  for (int point = 0; point < problem.PointCount(); ++point) {
    ordering->AddElementToGroup(points[point].data(), 0);
  }
  for (int camera = 0; camera < problem.CameraCount(); ++camera) {
    ordering->AddElementToGroup(cameras[camera].data(), 1);
  }

  for (const auto& observation : problem.observations) {
    ceres_problem.AddResidualBlock(new ReprojectionCost(observation.measurement), nullptr,
                                   cameras[observation.camera].data(),
                                   points[observation.point].data());
  }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::SPARSE_SCHUR;
  options.linear_solver_ordering = ordering;
  options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  options.dense_linear_algebra_library_type = ceres::EIGEN;
  options.jacobi_scaling = tiny_opts.lm.jacobi_scaling;
  options.initial_trust_region_radius = 1.0 / tiny_opts.lm.damping_init;
  options.max_num_iterations = tiny_opts.stop.max_iters;
  options.function_tolerance = tiny_opts.stop.min_rerr_dec;
  options.gradient_tolerance = tiny_opts.stop.min_grad_max;
  options.parameter_tolerance = std::sqrt(tiny_opts.stop.min_step_rel_norm2);
  options.min_relative_decrease = 1e-12;
  options.num_threads = 1;
  options.logging_type = ceres::SILENT;
  options.minimizer_progress_to_stdout = false;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &ceres_problem, &summary);
  return {summary.initial_cost,
          summary.final_cost,
          static_cast<int>(summary.iterations.size()) - 1,
          summary.termination_type == ceres::CONVERGENCE,
          summary.message,
          std::move(cameras),
          std::move(points)};
}

void RunBlockSparseBATest(double noise_sigma, double max_cam_tol, double max_pt_tol) {
  const auto options = MakeTinyoptLMOptions();
  const Problem problem = MakeProblem(100, 50000, noise_sigma);

  // 1. Solve with Ceres
  const auto ceres_result = SolveCeres(problem, options);
  REQUIRE(ceres_result.converged);

  // 2. Solve with Tinyopt
  Problem tinyopt_problem = problem;
  const BlockSparseBundleAdjustmentLoss loss;
  using BlockHessian = BlockSparseBundleAdjustmentLoss::Hessian;
  lm::Optimizer<BlockHessian> optimizer(options);
  const auto tinyopt_summary = optimizer(tinyopt_problem, loss);

  REQUIRE(tinyopt_summary.Succeeded());
  REQUIRE(tinyopt_summary.Converged());

  const int ceres_iters = ceres_result.iterations;
  const int tinyopt_iters = tinyopt_summary.num_iters;

  const double initial_cost = ReferenceReprojectionCost(problem);

  // Initial cost matches
  REQUIRE(initial_cost == Catch::Approx(ceres_result.initial_cost).epsilon(1e-8));

  // Final cost matches
  if (ceres_result.final_cost < 1e-5) {
    REQUIRE(tinyopt_summary.final_cost.cost < 1e-4);
  } else {
    REQUIRE(tinyopt_summary.final_cost.cost ==
            Catch::Approx(ceres_result.final_cost).epsilon(1e-4));
  }

  // Verify camera parameters match
  double max_cam_diff = 0.0;
  for (int c = 0; c < problem.CameraCount(); ++c) {
    const double cam_diff = (tinyopt_problem.initial_cameras[c] - ceres_result.cameras[c]).norm();
    max_cam_diff = std::max(max_cam_diff, cam_diff);
  }

  // Verify point parameters match
  double max_point_diff = 0.0;
  for (int p = 0; p < problem.PointCount(); ++p) {
    const double pt_diff = (tinyopt_problem.initial_points[p] - ceres_result.points[p]).norm();
    max_point_diff = std::max(max_point_diff, pt_diff);
  }

  INFO("BA 100c 50000p (noise=" << noise_sigma << ") - Ceres iters: " << ceres_iters
                                << ", Tinyopt iters: " << tinyopt_iters
                                << ", Ceres cost: " << ceres_result.final_cost
                                << ", Tinyopt cost: " << tinyopt_summary.final_cost.cost
                                << ", Max cam diff: " << max_cam_diff
                                << ", Max point diff: " << max_point_diff);

  REQUIRE(max_cam_diff < max_cam_tol);
  REQUIRE(max_point_diff < max_pt_tol);

  // Gauge parameters must remain fixed
  REQUIRE(tinyopt_problem.initial_cameras[0] == problem.initial_cameras[0]);
  REQUIRE(tinyopt_problem.initial_points[0] == problem.initial_points[0]);

  // Both solvers must converge in reasonable iterations
  REQUIRE(tinyopt_summary.Converged());
  REQUIRE(ceres_result.converged);
  REQUIRE(tinyopt_iters <= 100);
}

}  // namespace

TEST_CASE("exact_lm_block_sparse_ba_100c_50000p_noise_free",
          "[ceres_exact][lm][ba][block_sparse]") {
  RunBlockSparseBATest(0.0, 0.05, 0.05);
}

TEST_CASE("exact_lm_block_sparse_ba_100c_50000p_noisy", "[ceres_exact][lm][ba][block_sparse]") {
  RunBlockSparseBATest(0.5, 0.05, 0.05);
}
