// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <string>

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>

#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

#include "robust_pose.h"
#include "g2o_termination.h"
#include "iterations.h"

namespace {

g2o::SE3Quat ToPose(const tinyopt::benchmark::robust_pose::Pose& pose) {
  return {pose.unit_quaternion(), pose.translation()};
}

class PointResidual final : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, g2o::VertexSE3Expmap> {
 public:
  explicit PointResidual(const tinyopt::benchmark::robust_pose::Observation& observation)
      : world_point_(observation.world_point) {
    _measurement = observation.measurement;
  }

  void computeError() override {
    const auto* pose = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
    _error = pose->estimate().map(world_point_) - _measurement;
  }

  void linearizeOplus() override {
    const auto* pose = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
    const Eigen::Vector3d transformed = pose->estimate().map(world_point_);
    Eigen::Matrix3d skew;
    skew << 0, -transformed[2], transformed[1], transformed[2], 0, -transformed[0],
        -transformed[1], transformed[0], 0;
    _jacobianOplusXi.leftCols<3>() = -skew;
    _jacobianOplusXi.rightCols<3>().setIdentity();
  }

  bool read(std::istream&) override { return false; }
  bool write(std::ostream&) const override { return false; }

 private:
  Eigen::Vector3d world_point_;
};

struct Result {
  double initial_cost;
  double final_cost;
  int iterations;
  bool converged;
};

Result Solve(const std::vector<tinyopt::benchmark::robust_pose::Observation>& observations,
             const tinyopt::benchmark::robust_pose::Pose& initial_pose, bool robust) {
  using BlockSolver = g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>>;
  using LinearSolver = g2o::LinearSolverEigen<BlockSolver::PoseMatrixType>;
  g2o::SparseOptimizer optimizer;
  optimizer.setVerbose(false);
  auto linear_solver = std::make_unique<LinearSolver>();
  linear_solver->setBlockOrdering(false);
  auto block_solver = std::make_unique<BlockSolver>(std::move(linear_solver));
  auto* algorithm = new g2o::OptimizationAlgorithmLevenberg(std::move(block_solver));
  algorithm->setUserLambdaInit(1e-4);
  algorithm->setMaxTrialsAfterFailure(20);
  optimizer.setAlgorithm(algorithm);

  auto* pose = new g2o::VertexSE3Expmap();
  pose->setId(0);
  pose->setEstimate(ToPose(initial_pose));
  optimizer.addVertex(pose);

  for (const auto& observation : observations) {
    auto* edge = new PointResidual(observation);
    edge->setVertex(0, pose);
    edge->setInformation(Eigen::Matrix3d::Identity());
    if (robust) {
      auto* kernel = new g2o::RobustKernelHuber();
      kernel->setDelta(0.3);
      edge->setRobustKernel(kernel);
    }
    optimizer.addEdge(edge);
  }

  optimizer.initializeOptimization();
  tinyopt::benchmark::G2oTerminationAction termination(1000);
  optimizer.addPostIterationAction(&termination);
  optimizer.computeActiveErrors();
  const double initial_cost = 0.5 * optimizer.activeChi2();
  const int iterations = optimizer.optimize(1000);
  optimizer.computeActiveErrors();
  return {initial_cost, 0.5 * optimizer.activeChi2(), iterations, termination.Converged()};
}

void BenchmarkPose(int observation_count, bool robust) {
  using namespace tinyopt::benchmark::robust_pose;
  const Pose ground_truth = GroundTruth();
  const int outliers = observation_count / 4;
  const auto observations =
      MakeObservations(ground_truth, observation_count - outliers, outliers, 0.01, 50.0);
  const Result verification = Solve(observations, InitialPose(ground_truth), robust);
  REQUIRE(verification.iterations > 0);
  REQUIRE(verification.converged);
  tinyopt::benchmark::PrintIterations(
      "Robust pose", std::to_string(observation_count) + "o " + (robust ? "Huber" : "L2"),
      "g2o", verification.iterations, verification.converged);
  if (!robust) REQUIRE(verification.final_cost < verification.initial_cost);
  BENCHMARK(std::string(std::to_string(observation_count) + " obs " + (robust ? "Huber" : "L2"))) {
    return Solve(observations, InitialPose(ground_truth), robust).final_cost;
  };
}

}  // namespace

TEST_CASE("PoseOptimization", "[benchmark][robust][pose][g2o]") {
  for (const int count : {20, 50, 100}) {
    BenchmarkPose(count, false);
    BenchmarkPose(count, true);
  }
}
