// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>

#include "sparse_problem.h"
#include "g2o_termination.h"
#include "iterations.h"

namespace {

class ScalarVertex final : public g2o::BaseVertex<1, double> {
 public:
  void setToOriginImpl() override { _estimate = 0; }
  void oplusImpl(const double* update) override { _estimate += update[0]; }
  int minimalEstimateDimension() const override { return 1; }
  bool getMinimalEstimateData(double* estimate) const override {
    estimate[0] = _estimate;
    return true;
  }
  bool read(std::istream& stream) override { return static_cast<bool>(stream >> _estimate); }
  bool write(std::ostream& stream) const override { return static_cast<bool>(stream << _estimate); }
};

class UnaryResidual final : public g2o::BaseUnaryEdge<1, double, ScalarVertex> {
 public:
  void computeError() override {
    _error[0] = static_cast<const ScalarVertex*>(_vertices[0])->estimate() - _measurement;
  }

  void linearizeOplus() override { _jacobianOplusXi[0] = 1; }

  bool read(std::istream& stream) override { return static_cast<bool>(stream >> _measurement); }

  bool write(std::ostream& stream) const override {
    return static_cast<bool>(stream << _measurement);
  }
};

class PairResidual final : public g2o::BaseBinaryEdge<1, double, ScalarVertex, ScalarVertex> {
 public:
  void computeError() override {
    const auto* first = static_cast<const ScalarVertex*>(_vertices[0]);
    const auto* second = static_cast<const ScalarVertex*>(_vertices[1]);
    _error[0] = 0.1 * ((second->estimate() - first->estimate()) - _measurement);
  }

  void linearizeOplus() override {
    _jacobianOplusXi[0] = -0.1;
    _jacobianOplusXj[0] = 0.1;
  }

  bool read(std::istream& stream) override { return static_cast<bool>(stream >> _measurement); }

  bool write(std::ostream& stream) const override {
    return static_cast<bool>(stream << _measurement);
  }
};

struct Result {
  double final_cost;
  std::vector<double> parameters;
  int iterations;
  bool converged;
};

Result Solve(int dimensions) {
  using BlockSolver = g2o::BlockSolver<g2o::BlockSolverTraits<1, 1>>;
  using LinearSolver = g2o::LinearSolverEigen<BlockSolver::PoseMatrixType>;
  g2o::SparseOptimizer optimizer;
  optimizer.setVerbose(false);
  auto linear_solver = std::make_unique<LinearSolver>();
  linear_solver->setBlockOrdering(false);
  auto block_solver = std::make_unique<BlockSolver>(std::move(linear_solver));
  auto* algorithm = new g2o::OptimizationAlgorithmLevenberg(std::move(block_solver));
  algorithm->setUserLambdaInit(1e-4);
  algorithm->setMaxTrialsAfterFailure(3);
  optimizer.setAlgorithm(algorithm);

  std::vector<ScalarVertex*> vertices(dimensions);
  for (int index = 0; index < dimensions; ++index) {
    auto* vertex = new ScalarVertex();
    vertex->setId(index);
    vertex->setEstimate(tinyopt::benchmark::sparse_problem::Initial(index));
    optimizer.addVertex(vertex);
    vertices[index] = vertex;

    auto* unary = new UnaryResidual();
    unary->setVertex(0, vertex);
    unary->setMeasurement(tinyopt::benchmark::sparse_problem::Target(index));
    unary->setInformation(Eigen::Matrix<double, 1, 1>::Identity());
    optimizer.addEdge(unary);

    if (index > 0) {
      auto* pair = new PairResidual();
      pair->setVertex(0, vertices[index - 1]);
      pair->setVertex(1, vertex);
      pair->setMeasurement(tinyopt::benchmark::sparse_problem::DifferenceTarget(index));
      pair->setInformation(Eigen::Matrix<double, 1, 1>::Identity());
      optimizer.addEdge(pair);
    }
  }

  optimizer.initializeOptimization();
  tinyopt::benchmark::G2oTerminationAction termination(100);
  optimizer.addPostIterationAction(&termination);
  optimizer.computeActiveErrors();
  const int iterations = optimizer.optimize(100);
  optimizer.computeActiveErrors();
  Result result{0.5 * optimizer.activeChi2(), {}, iterations, termination.Converged()};
  result.parameters.reserve(dimensions);
  for (const auto* vertex : vertices) result.parameters.push_back(vertex->estimate());
  return result;
}

}  // namespace

TEST_CASE("Sparse", "[benchmark][sparse][g2o]") {
  const int dimensions = GENERATE(10, 100, 1000);
  CAPTURE(dimensions);
  const Result result = Solve(dimensions);
  REQUIRE(result.converged);
  REQUIRE(result.final_cost < 1e-10);
  for (int index = 0; index < dimensions; ++index)
    REQUIRE(result.parameters[index] ==
            Catch::Approx(tinyopt::benchmark::sparse_problem::Target(index)).margin(1e-6));
  tinyopt::benchmark::PrintIterations("Sparse", std::to_string(dimensions) + "d", "g2o",
                                      result.iterations, result.converged);
  BENCHMARK(std::to_string(dimensions) + "D sparse chain") { return Solve(dimensions).final_cost; };
}
