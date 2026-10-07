// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <string>

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <ceres/ceres.h>

#include "dense_problems.h"
#include "iterations.h"

namespace {

ceres::Solver::Options MakeOptions() {
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
  options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  options.dense_linear_algebra_library_type = ceres::EIGEN;
  options.max_num_iterations = 100;
  options.max_num_consecutive_invalid_steps = 3;
  options.num_threads = 1;
  options.function_tolerance = 1e-6;
  options.gradient_tolerance = 1e-9;
  options.parameter_tolerance = 1e-8;
  options.min_relative_decrease = 1e-12;
  options.logging_type = ceres::SILENT;
  return options;
}

struct Result {
  double cost;
  int iterations;
  bool converged;
};

template <int Dimensions>
class FixedMathCost final : public ceres::SizedCostFunction<Dimensions, Dimensions> {
 public:
  bool Evaluate(double const* const* parameters, double* residuals,
                double** jacobians) const override {
    const Eigen::Map<const Eigen::Matrix<double, Dimensions, 1>> x(parameters[0]);
    Eigen::Map<Eigen::Matrix<double, Dimensions, 1>> output(residuals);
    output = tinyopt::benchmark::DenseMathResiduals(x);
    if (jacobians != nullptr && jacobians[0] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, Dimensions, Dimensions, Eigen::RowMajor>> J(jacobians[0]);
      tinyopt::benchmark::DenseMathJacobian(x, J);
    }
    return true;
  }
};

class DynamicMathCost final : public ceres::CostFunction {
 public:
  explicit DynamicMathCost(int dimensions) : dimensions_(dimensions) {
    set_num_residuals(dimensions_);
    mutable_parameter_block_sizes()->push_back(dimensions_);
  }

  bool Evaluate(double const* const* parameters, double* residuals,
                double** jacobians) const override {
    const Eigen::Map<const Eigen::VectorXd> x(parameters[0], dimensions_);
    Eigen::Map<Eigen::VectorXd> output(residuals, dimensions_);
    output = tinyopt::benchmark::DenseMathResiduals(x);
    if (jacobians != nullptr && jacobians[0] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> J(
          jacobians[0], dimensions_, dimensions_);
      tinyopt::benchmark::DenseMathJacobian(x, J);
    }
    return true;
  }

 private:
  int dimensions_;
};

class DynamicPriorCost final : public ceres::CostFunction {
 public:
  explicit DynamicPriorCost(Eigen::Index dimensions)
      : target_(tinyopt::benchmark::PriorTarget<double>(dimensions)) {
    set_num_residuals(static_cast<int>(dimensions));
    mutable_parameter_block_sizes()->push_back(static_cast<int>(dimensions));
  }

  bool Evaluate(double const* const* parameters, double* residuals,
                double** jacobians) const override {
    const Eigen::Map<const Eigen::VectorXd> x(parameters[0], target_.size());
    Eigen::Map<Eigen::VectorXd> output(residuals, target_.size());
    output = x - target_;
    if (jacobians != nullptr && jacobians[0] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> J(
          jacobians[0], target_.size(), target_.size());
      J.setIdentity();
    }
    return true;
  }

 private:
  Eigen::VectorXd target_;
};

Result Solve(ceres::CostFunction* cost, double* parameters) {
  ceres::Problem problem;
  problem.AddResidualBlock(cost, nullptr, parameters);
  ceres::Solver::Summary summary;
  ceres::Solve(MakeOptions(), &problem, &summary);
  return {summary.final_cost, static_cast<int>(summary.iterations.size()) - 1,
          summary.termination_type == ceres::CONVERGENCE};
}

template <int Dimensions>
Result SolveFixed(Eigen::Matrix<double, Dimensions, 1>& x) {
  return Solve(new FixedMathCost<Dimensions>(), x.data());
}

Result SolveDynamic(Eigen::VectorXd& x, bool prior) {
  return Solve(prior ? static_cast<ceres::CostFunction*>(new DynamicPriorCost(x.size()))
                     : static_cast<ceres::CostFunction*>(new DynamicMathCost(x.size())),
               x.data());
}

template <int Dimensions>
void BenchmarkFixedMath() {
  Eigen::Matrix<double, Dimensions, 1> x = tinyopt::benchmark::DenseMathInitial<double>(Dimensions);
  const Result result = SolveFixed<Dimensions>(x);
  REQUIRE(result.converged);
  REQUIRE(result.cost < 1e-12);
  if constexpr (Dimensions == 1)
    REQUIRE(x[0] == Catch::Approx(std::sqrt(2.0)).epsilon(1e-7));
  else
    REQUIRE((x.array() - 1.0).matrix().norm() < 1e-6);
  tinyopt::benchmark::PrintIterations("Dense static", std::to_string(Dimensions) + "d",
                                      "ceres", result.iterations, result.converged);
  BENCHMARK(std::to_string(Dimensions) + "D static double") {
    Eigen::Matrix<double, Dimensions, 1> parameters =
        tinyopt::benchmark::DenseMathInitial<double>(Dimensions);
    return SolveFixed<Dimensions>(parameters).cost;
  };
}

void BenchmarkDynamicMath(Eigen::Index dimensions) {
  Eigen::VectorXd x = tinyopt::benchmark::DenseMathInitial<double>(dimensions);
  const Result result = SolveDynamic(x, false);
  REQUIRE(result.converged);
  REQUIRE(result.cost < 1e-12);
  if (dimensions == 1)
    REQUIRE(x[0] == Catch::Approx(std::sqrt(2.0)).epsilon(1e-7));
  else
    REQUIRE((x.array() - 1.0).matrix().norm() < 1e-6);
  tinyopt::benchmark::PrintIterations("Dense dynamic", std::to_string(dimensions) + "d",
                                      "ceres", result.iterations, result.converged);
  BENCHMARK(std::to_string(dimensions) + "D dynamic double") {
    Eigen::VectorXd parameters = tinyopt::benchmark::DenseMathInitial<double>(dimensions);
    return SolveDynamic(parameters, false).cost;
  };
}

void BenchmarkDynamicPrior(Eigen::Index dimensions) {
  Eigen::VectorXd x = tinyopt::benchmark::PriorInitial<double>(dimensions);
  const Eigen::VectorXd target = tinyopt::benchmark::PriorTarget<double>(dimensions);
  const Result result = SolveDynamic(x, true);
  REQUIRE(result.converged);
  REQUIRE(result.cost < 1e-12);
  REQUIRE((x - target).norm() < 1e-7);
  tinyopt::benchmark::PrintIterations("Dense dynamic", std::to_string(dimensions) + "dp",
                                      "ceres", result.iterations, result.converged);
  BENCHMARK(std::to_string(dimensions) + "D dynamic double prior") {
    Eigen::VectorXd parameters = tinyopt::benchmark::PriorInitial<double>(dimensions);
    return SolveDynamic(parameters, true).cost;
  };
}

}  // namespace

TEST_CASE("Dense", "[benchmark][dense][ceres]") {
  BenchmarkFixedMath<1>();
  BenchmarkFixedMath<2>();
  BenchmarkFixedMath<3>();
  for (const Eigen::Index dimensions : {1, 2, 3}) BenchmarkDynamicMath(dimensions);
  for (const Eigen::Index dimensions : {6, 12, 33, 50}) BenchmarkDynamicPrior(dimensions);
}
