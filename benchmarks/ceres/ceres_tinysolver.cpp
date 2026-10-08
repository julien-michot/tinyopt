// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <string>
#include <type_traits>

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <ceres/tiny_solver.h>

#include "dense_problems.h"
#include "iterations.h"

namespace {

template <typename T, int Dimensions>
struct MathFunction {
  using Scalar = T;
  using Parameters = Eigen::Matrix<T, Dimensions, 1>;
  enum { NUM_RESIDUALS = Dimensions, NUM_PARAMETERS = Dimensions };

  bool operator()(const T* parameters, T* residuals, T* jacobian) const {
    Eigen::Map<const Parameters> x(parameters);
    const auto values = tinyopt::benchmark::DenseMathResiduals(x);
    Eigen::Map<Eigen::Matrix<T, Dimensions, 1>> residual_map(residuals);
    residual_map = values;
    if (jacobian == nullptr) return true;

    Eigen::Map<Eigen::Matrix<T, Dimensions, Dimensions>> J(jacobian);
    J.setZero();
    if constexpr (Dimensions == 1) {
      J(0, 0) = T(2) * x[0];
    } else if constexpr (Dimensions == 2) {
      J(0, 0) = T(2) * x[0];
      J(0, 1) = T(1);
      J(1, 0) = T(1);
      J(1, 1) = T(2) * x[1];
    } else {
      J.setOnes();
      J.diagonal() = T(2) * x;
    }
    return true;
  }
};

template <typename T>
struct DynamicMathFunction {
  using Scalar = T;
  enum { NUM_RESIDUALS = Eigen::Dynamic, NUM_PARAMETERS = Eigen::Dynamic };

  int NumResiduals() const { return dimensions; }
  int NumParameters() const { return dimensions; }

  bool operator()(const T* parameters, T* residuals, T* jacobian) const {
    const Eigen::Map<const Eigen::Vector<T, Eigen::Dynamic>> x(parameters, dimensions);
    Eigen::Map<Eigen::Vector<T, Eigen::Dynamic>> residual_map(residuals, dimensions);
    residual_map = tinyopt::benchmark::DenseMathResiduals(x);
    if (jacobian == nullptr) return true;

    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> J(jacobian, dimensions,
                                                                   dimensions);
    J.setZero();
    if (dimensions == 1) {
      J(0, 0) = T(2) * x[0];
    } else if (dimensions == 2) {
      J(0, 0) = T(2) * x[0];
      J(0, 1) = T(1);
      J(1, 0) = T(1);
      J(1, 1) = T(2) * x[1];
    } else {
      J.setOnes();
      J.diagonal() = T(2) * x;
    }
    return true;
  }

  int dimensions;
};

template <typename Scalar, int Dimensions>
using Solver = ceres::TinySolver<MathFunction<Scalar, Dimensions>>;

template <typename Scalar, int Dimensions>
void RunCase() {
  using Vector = Eigen::Matrix<Scalar, Dimensions, 1>;
  Solver<Scalar, Dimensions> solver;
  solver.options.max_num_iterations = 100;
  solver.options.gradient_tolerance = std::is_same_v<Scalar, float> ? 1e-5f : 1e-9;
  solver.options.parameter_tolerance = std::is_same_v<Scalar, float> ? 1e-5f : 1e-8;
  solver.options.function_tolerance = 1e-6;

  Vector parameters = tinyopt::benchmark::DenseMathInitial<Scalar>(Dimensions);
  const auto& summary = solver.Solve(MathFunction<Scalar, Dimensions>(), &parameters);
  REQUIRE(summary.status != Solver<Scalar, Dimensions>::HIT_MAX_ITERATIONS);
  REQUIRE(summary.final_cost < (std::is_same_v<Scalar, float> ? 1e-8f : 1e-12));
  if constexpr (Dimensions == 1)
    REQUIRE(parameters[0] == Catch::Approx(std::sqrt(Scalar(2))).epsilon(1e-4));
  else
    REQUIRE((parameters.array() - Scalar(1)).matrix().norm() <
            (std::is_same_v<Scalar, float> ? 1e-4f : 1e-6));

  const std::string type = std::is_same_v<Scalar, float> ? "f" : "d";
  const std::string precision = std::is_same_v<Scalar, float> ? "float" : "double";
  tinyopt::benchmark::PrintIterations(
      "Dense static", std::to_string(Dimensions) + type, "ceres-tinysolver", summary.iterations,
      summary.status != Solver<Scalar, Dimensions>::HIT_MAX_ITERATIONS);
  BENCHMARK(std::to_string(Dimensions) + "D static " + precision) {
    Vector x = tinyopt::benchmark::DenseMathInitial<Scalar>(Dimensions);
    const auto& result = solver.Solve(MathFunction<Scalar, Dimensions>(), &x);
    return result.final_cost;
  };
}

template <typename Scalar>
void RunDynamicCase(int dimensions) {
  using Function = DynamicMathFunction<Scalar>;
  using Solver = ceres::TinySolver<Function>;
  Solver solver;
  solver.options.max_num_iterations = 100;
  solver.options.gradient_tolerance = std::is_same_v<Scalar, float> ? 1e-5f : 1e-9;
  solver.options.parameter_tolerance = std::is_same_v<Scalar, float> ? 1e-5f : 1e-8;
  solver.options.function_tolerance = 1e-6;

  Eigen::Vector<Scalar, Eigen::Dynamic> parameters =
      tinyopt::benchmark::DenseMathInitial<Scalar>(dimensions);
  const Function function{dimensions};
  const auto& summary = solver.Solve(function, &parameters);
  REQUIRE(summary.status != Solver::HIT_MAX_ITERATIONS);
  REQUIRE(summary.final_cost < (std::is_same_v<Scalar, float> ? 1e-8f : 1e-12));
  if (dimensions == 1)
    REQUIRE(std::abs(parameters[0] - std::sqrt(Scalar(2))) <
            (std::is_same_v<Scalar, float> ? 1e-4f : 1e-6));
  else
    REQUIRE((parameters.array() - Scalar(1)).matrix().norm() <
            (std::is_same_v<Scalar, float> ? 1e-4f : 1e-6));

  const std::string type = std::is_same_v<Scalar, float> ? "f" : "d";
  tinyopt::benchmark::PrintIterations("Dense dynamic", std::to_string(dimensions) + type,
                                      "ceres-tinysolver", summary.iterations,
                                      summary.status != Solver::HIT_MAX_ITERATIONS);
  const std::string precision = std::is_same_v<Scalar, float> ? "float" : "double";
  BENCHMARK(std::string(std::to_string(dimensions) + "D dynamic " + precision)) {
    Eigen::Vector<Scalar, Eigen::Dynamic> x =
        tinyopt::benchmark::DenseMathInitial<Scalar>(dimensions);
    const auto& result = solver.Solve(function, &x);
    return result.final_cost;
  };
}

}  // namespace

TEST_CASE("Dense", "[benchmark][dense][ceres-tinysolver]") {
  RunCase<double, 1>();
  RunCase<double, 2>();
  RunCase<double, 3>();
  RunCase<float, 1>();
  RunCase<float, 2>();
  RunCase<float, 3>();
  for (int dimensions = 1; dimensions <= 3; ++dimensions) {
    RunDynamicCase<double>(dimensions);
    RunDynamicCase<float>(dimensions);
  }
}
