// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <string>

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

#include "sparse_problem.h"
#include "iterations.h"

namespace {

gtsam::Key Key(int index) { return gtsam::Symbol('x', index); }

class UnaryResidual final : public gtsam::NoiseModelFactor1<double> {
 public:
  UnaryResidual(gtsam::Key key, double target, const gtsam::SharedNoiseModel& noise)
      : Base(noise, key), target_(target) {}

  gtsam::Vector evaluateError(const double& value,
                              gtsam::OptionalMatrixType jacobian = OptionalNone) const override {
    if (jacobian != nullptr) *jacobian = gtsam::Matrix::Ones(1, 1);
    return gtsam::Vector1(value - target_);
  }

 private:
  using Base = gtsam::NoiseModelFactor1<double>;
  double target_;
};

class PairResidual final : public gtsam::NoiseModelFactor2<double, double> {
 public:
  PairResidual(gtsam::Key first, gtsam::Key second, double target_difference,
               const gtsam::SharedNoiseModel& noise)
      : Base(noise, first, second), target_difference_(target_difference) {}

  gtsam::Vector evaluateError(
      const double& first, const double& second,
      gtsam::OptionalMatrixType first_jacobian = OptionalNone,
      gtsam::OptionalMatrixType second_jacobian = OptionalNone) const override {
    if (first_jacobian != nullptr) *first_jacobian = gtsam::Matrix::Constant(1, 1, -0.1);
    if (second_jacobian != nullptr) *second_jacobian = gtsam::Matrix::Constant(1, 1, 0.1);
    return gtsam::Vector1(0.1 * ((second - first) - target_difference_));
  }

 private:
  using Base = gtsam::NoiseModelFactor2<double, double>;
  double target_difference_;
};

struct Result {
  double final_cost;
  gtsam::Values values;
  int iterations;
};

Result Solve(int dimensions) {
  gtsam::NonlinearFactorGraph graph;
  gtsam::Values initial;
  const auto noise = gtsam::noiseModel::Isotropic::Sigma(1, 1.0);
  for (int index = 0; index < dimensions; ++index) {
    initial.insert(Key(index), tinyopt::benchmark::sparse_problem::Initial(index));
    graph.emplace_shared<UnaryResidual>(Key(index),
                                        tinyopt::benchmark::sparse_problem::Target(index), noise);
    if (index > 0) {
      graph.emplace_shared<PairResidual>(
          Key(index - 1), Key(index), tinyopt::benchmark::sparse_problem::DifferenceTarget(index),
          noise);
    }
  }

  gtsam::LevenbergMarquardtParams options = gtsam::LevenbergMarquardtParams::CeresDefaults();
  options.maxIterations = 100;
  options.relativeErrorTol = 1e-6;
  options.absoluteErrorTol = 0;
  options.errorTol = 1e-12;
  options.lambdaInitial = 1e-4;
  options.minModelFidelity = 1e-12;
  options.verbosityLM = gtsam::LevenbergMarquardtParams::SILENT;
  options.linearSolverType = gtsam::NonlinearOptimizerParams::MULTIFRONTAL_CHOLESKY;

  gtsam::LevenbergMarquardtOptimizer optimizer(graph, initial, options);
  const auto result = optimizer.optimize();
  return {graph.error(result), result, static_cast<int>(optimizer.iterations())};
}

}  // namespace

TEST_CASE("Sparse", "[benchmark][sparse][gtsam]") {
  const int dimensions = GENERATE(10, 100, 1000);
  CAPTURE(dimensions);
  const Result result = Solve(dimensions);
  REQUIRE(result.iterations < 100);
  REQUIRE(result.final_cost < 1e-10);
  for (int index = 0; index < dimensions; ++index)
    REQUIRE(result.values.at<double>(Key(index)) ==
            Catch::Approx(tinyopt::benchmark::sparse_problem::Target(index)).margin(1e-6));
  tinyopt::benchmark::PrintIterations("Sparse", std::to_string(dimensions) + "d", "gtsam",
                                      result.iterations, result.iterations < 100);
  BENCHMARK(std::to_string(dimensions) + "D sparse chain") { return Solve(dimensions).final_cost; };
}
