// Copyright (C) 2025 Julien Michot. All Rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cmath>
<<<<<<< HEAD
#include "tinyopt/math.h"
=======
>>>>>>> 06ca573 (Add Ceres test)

#if CATCH2_VERSION == 2
#include <catch2/catch.hpp>
#else
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#endif

#include <Eigen/Core>

#include <ceres/ceres.h>
#include <tinyopt/losses/mahalanobis.h>

using namespace tinyopt;
using namespace tinyopt::losses;

class Sqrt2CostFunctor {
 public:
  template <typename T>
  bool operator()(const T* const x, T* residual) const {
    residual[0] = x[0] * x[0] - T(2.0);
    return true;
  }
};

inline auto CreateOptions() {
  ceres::Solver::Options options;
  options.minimizer_progress_to_stdout = true;
  options.max_num_iterations = 20;
  options.check_gradients = false;
  options.parameter_tolerance = 1e-8;
  options.function_tolerance = 1e-6;
  options.min_relative_decrease = 1e-3;
  options.gradient_tolerance = 1e-10;
  options.gradient_check_relative_precision = 1e-8;
  options.gradient_check_numeric_derivative_relative_step_size = 1e-6;
  options.linear_solver_type = ceres::LinearSolverType::DENSE_NORMAL_CHOLESKY;
  options.trust_region_strategy_type = ceres::TrustRegionStrategyType::LEVENBERG_MARQUARDT;
  options.dense_linear_algebra_library_type = ceres::DenseLinearAlgebraLibraryType::EIGEN;
  return options;
}

TEST_CASE("Scalar") {
  const auto options = CreateOptions();

<<<<<<< HEAD
  double x = GENERATE(1.0, -0.3, 3.2);
  CAPTURE(x);

  SECTION("√2") {
=======
  SECTION("√2") {
    double x = -0.3;
>>>>>>> 06ca573 (Add Ceres test)
    std::cout << "x:" << x << "\n";
    ceres::Problem problem;
    problem.AddParameterBlock(&x, 1);  // Optimize the single variable 'x'
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<Sqrt2CostFunctor, 1, 1>(  // Use AutoDiffCostFunction
            new Sqrt2CostFunctor),
        nullptr,                     // No loss function.
        &x);                         // The parameter block to which the cost function applies.
    ceres::Solver::Summary summary;  // Summary of the optimization.
    ceres::Solve(options, &problem, &summary);  // Solve the problem!
<<<<<<< HEAD
    REQUIRE(std::abs(x) == Catch::Approx(std::sqrt(2.0)).margin(1e-5));
=======
    std::cout << summary.FullReport() << "\n";  // Detailed report.
    std::cout << "final x:" << x << "\n";
>>>>>>> 06ca573 (Add Ceres test)
  };
}
