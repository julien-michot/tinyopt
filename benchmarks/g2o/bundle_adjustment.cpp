// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

#include "bundle_adjustment.h"
#include "g2o_termination.h"
#include "iterations.h"

using namespace tinyopt::benchmark::bundle_adjustment;

namespace {

struct Result {
  double initial_cost;
  double final_cost;
  int iterations;
  bool converged;
};

g2o::SE3Quat ToPose(const Camera& camera) {
  Eigen::Matrix3d rotation;
  const Eigen::Vector3d angles = camera.head<3>();
  for (int column = 0; column < 3; ++column) {
    Eigen::Vector3d basis = Eigen::Vector3d::Zero();
    basis[column] = 1.0;
    rotation.col(column) = RotateEuler(angles, basis);
  }
  return {Eigen::Quaterniond(rotation), camera.tail<3>()};
}

Result Optimize(const Problem& problem) {
  using BlockSolver = g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>>;
  using LinearSolver = g2o::LinearSolverEigen<BlockSolver::PoseMatrixType>;

  g2o::SparseOptimizer optimizer;
  optimizer.setVerbose(false);
  auto linear_solver = std::make_unique<LinearSolver>();
  linear_solver->setBlockOrdering(true);
  auto block_solver = std::make_unique<BlockSolver>(std::move(linear_solver));
  block_solver->setSchur(true);
  auto* algorithm = new g2o::OptimizationAlgorithmLevenberg(std::move(block_solver));
  algorithm->setUserLambdaInit(1e-4);
  algorithm->setMaxTrialsAfterFailure(3);
  optimizer.setAlgorithm(algorithm);

  auto* calibration =
      new g2o::CameraParameters(FocalX, Eigen::Vector2d(PrincipalX, PrincipalY), 0.0);
  calibration->setId(0);
  optimizer.addParameter(calibration);

  std::vector<g2o::VertexSE3Expmap*> cameras(problem.CameraCount());
  for (int camera = 0; camera < problem.CameraCount(); ++camera) {
    auto* vertex = new g2o::VertexSE3Expmap();
    vertex->setId(camera);
    vertex->setEstimate(ToPose(problem.initial_cameras[camera]));
    vertex->setFixed(camera == 0);
    optimizer.addVertex(vertex);
    cameras[camera] = vertex;
  }

  std::vector<g2o::VertexSBAPointXYZ*> points(problem.PointCount());
  for (int point = 0; point < problem.PointCount(); ++point) {
    auto* vertex = new g2o::VertexSBAPointXYZ();
    vertex->setId(problem.CameraCount() + point);
    vertex->setEstimate(problem.initial_points[point]);
    vertex->setFixed(point == 0);
    vertex->setMarginalized(true);
    optimizer.addVertex(vertex);
    points[point] = vertex;
  }

  for (const auto& observation : problem.observations) {
    auto* edge = new g2o::EdgeProjectXYZ2UV();
    edge->setVertex(0, points[observation.point]);
    edge->setVertex(1, cameras[observation.camera]);
    edge->setMeasurement(observation.measurement);
    edge->setInformation(Eigen::Matrix2d::Identity());
    edge->setParameterId(0, 0);
    optimizer.addEdge(edge);
  }

  optimizer.initializeOptimization();
  tinyopt::benchmark::G2oTerminationAction termination(100);
  optimizer.addPostIterationAction(&termination);
  optimizer.computeActiveErrors();
  const double initial_cost = 0.5 * optimizer.activeChi2();
  const int iterations = optimizer.optimize(100);
  optimizer.computeActiveErrors();
  return {initial_cost, 0.5 * optimizer.activeChi2(), iterations, termination.Converged()};
}

}  // namespace

TEST_CASE("BA", "[benchmark][bundle-adjustment][g2o]") {
  const auto dimensions = GENERATE(std::pair{5, 50}, std::pair{20, 200}, std::pair{50, 500});
  const Problem problem = MakeProblem(dimensions.first, dimensions.second);
  const double reference_initial_cost = ReferenceReprojectionCost(problem);
  const Result verification = Optimize(problem);
  REQUIRE(verification.iterations > 0);
  REQUIRE(verification.converged);
  REQUIRE(verification.initial_cost == Catch::Approx(reference_initial_cost).margin(1e-8));
  REQUIRE(verification.final_cost < verification.initial_cost * 1e-4);
  REQUIRE(verification.final_cost < 1e-5);
  tinyopt::benchmark::PrintIterations("Bundle adjustment", ProblemLabel(problem), "g2o",
                                      verification.iterations, verification.converged);

  const std::string label = ProblemLabel(problem);
  BENCHMARK(std::string(label)) { return Optimize(problem).final_cost; };
}