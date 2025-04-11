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

#if CATCH2_VERSION == 2
#include <catch2/catch.hpp>
#else
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#endif

#include <Eigen/Core>

#include <ceres/ceres.h>
#include <tinyopt/losses/mahalanobis.h>

using namespace tinyopt::losses;

class Sqrt2CostFunctor {
 public:
  template <typename T>
  bool operator()(const T* const x, T* residual) const {
    residual[0] = x[0] * x[0] - T(2.0);
    return true;
  }
};

auto CreateOptions() {
  ceres::Solver::Options options;
  options.minimizer_progress_to_stdout = false;
  options.max_num_iterations = 10;
  options.check_gradients = false;
  // TODO get options close to what Tinyopt is using
  options.function_tolerance = 1e-6;
  options.gradient_tolerance = 1e-10;
  options.parameter_tolerance = 1e-8;
  options.gradient_check_relative_precision = 1e-8;
  options.gradient_check_numeric_derivative_relative_step_size = 1e-6;
  options.linear_solver_type = ceres::LinearSolverType::DENSE_NORMAL_CHOLESKY;
  // options.sparse_linear_algebra_library_type =
  // ceres::SparseLinearAlgebraLibraryType::EIGEN_SPARSE;
  options.trust_region_strategy_type = ceres::TrustRegionStrategyType::LEVENBERG_MARQUARDT;
  options.dense_linear_algebra_library_type = ceres::DenseLinearAlgebraLibraryType::EIGEN;
  return options;
}

TEST_CASE("Scalar", "[benchmark][fixed][scalar]") {
  const auto options = CreateOptions();

  BENCHMARK("√2") {
    double x = Eigen::Vector<double, 1>::Random()[0];
    ceres::Problem problem;
    problem.AddParameterBlock(&x, 1);  // Optimize the single variable 'x'
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<Sqrt2CostFunctor, 1, 1>(  // Use AutoDiffCostFunction
            new Sqrt2CostFunctor),
        nullptr,                     // No loss function.
        &x);                         // The parameter block to which the cost function applies.
    ceres::Solver::Summary summary;  // Summary of the optimization.
    ceres::Solve(options, &problem, &summary);  // Solve the problem!
    // std::cout << summary.FullReport() << "\n";  // Detailed report.
  };
}

template <typename Vec>
class MahalanobisCostFunctor {
 public:
  static constexpr auto Dims = Vec::RowsAtCompileTime;
  MahalanobisCostFunctor(const Vec& prior,
                         const Vec& stdevs = Vec::Random(Dims == Eigen::Dynamic ? 10 : Dims))
      : prior_(prior), stdevs_{stdevs} {}

  template <typename T>
  bool operator()(T const* const* parameters, T* residuals) const {
    Eigen::Map<const Eigen::Vector<T, Dims>> x(parameters[0], prior_.size());
    Eigen::Map<Eigen::Vector<T, Dims>> r(residuals, prior_.size());

    // Convert prior_ to the same type T
    Eigen::Vector<T, Dims> prior_T = prior_.template cast<T>();
    Eigen::Vector<T, Dims> stdevs_T = stdevs_.template cast<T>();

    const auto delta = (x - prior_T).eval();
    r = MahaWhitened(delta, stdevs_T);
    return true;
  }

  template <typename T>
  bool operator()(T const* const parameters, T* residuals) const {
    Eigen::Map<const Eigen::Vector<T, Dims>> x(parameters, prior_.size());
    Eigen::Map<Eigen::Vector<T, Dims>> r(residuals, prior_.size());

    // Convert prior_ to the same type T
    Eigen::Vector<T, Dims> prior_T = prior_.template cast<T>();
    Eigen::Vector<T, Dims> stdevs_T = stdevs_.template cast<T>();

    const auto delta = (x - prior_T).eval();
    r = MahaWhitened(delta, stdevs_T);
    return true;
  }

  const Vec prior_;
  const Vec stdevs_;
};

template <typename Vec>
class MahalanobisFixedCostFunctor
    : public ceres::SizedCostFunction<Vec::RowsAtCompileTime, Vec::RowsAtCompileTime> {
 public:
  static constexpr auto Dims = Vec::RowsAtCompileTime;
  MahalanobisFixedCostFunctor(const Vec& prior, const Vec& stdevs = Vec::Random())
      : prior_(prior), stdevs_{stdevs} {}

  bool Evaluate(double const* const* parameters, double* residuals,
                double** jacobians) const override {
    Eigen::Map<const Vec> x(parameters[0], prior_.size());
    Eigen::Map<Vec> r(residuals, prior_.size());
    const auto delta = (x - prior_).eval();
    if (jacobians == nullptr) {
      r = MahaWhitened(delta, stdevs_);
    } else {
      using Mat = Eigen::Matrix<double, Dims, Dims>;
      Eigen::Map<Mat> jac(jacobians[0], prior_.size(), prior_.size());
      const auto& [res, J] = MahaWhitened(delta, stdevs_, true);
      r = res;
      jac = J;
    }
    return true;
  }

  const Vec prior_;
  const Vec stdevs_;
};

template <typename Vec>
class MahalanobisDynCostFunctor : public ceres::CostFunction {
 public:
  static constexpr auto Dims = Eigen::Dynamic;
  MahalanobisDynCostFunctor(const Vec& prior, const Vec& stdevs = Vec::Random(10))
      : prior_(prior), stdevs_{stdevs} {
    this->mutable_parameter_block_sizes()->push_back(prior.size());
    this->set_num_residuals(stdevs.size());
  }

  bool Evaluate(double const* const* parameters, double* residuals,
                double** jacobians) const override {
    Eigen::Map<const Vec> x(parameters[0], prior_.size());
    Eigen::Map<Vec> r(residuals, prior_.size());
    const auto delta = (x - prior_).eval();
    if (jacobians == nullptr) {
      r = MahaWhitened(delta, stdevs_);
    } else {
      using Mat = Eigen::Matrix<double, Dims, Dims>;
      Eigen::Map<Mat> jac(jacobians[0], prior_.size(), prior_.size());
      const auto& [res, J] = MahaWhitened(delta, stdevs_, true);
      r = res;
      jac = J;
    }
    return true;
  }

  const Vec prior_;
  const Vec stdevs_;
};

TEMPLATE_TEST_CASE("Dense", "[benchmark][fixed][dense][double]", tinyopt::Vec3, tinyopt::Vec6,
                   tinyopt::VecX) {
  constexpr tinyopt::Index Dims = TestType::RowsAtCompileTime;
  const int dims = Dims == Eigen::Dynamic ? 10 : Dims;
  const TestType y = TestType::Random(dims);

  static_assert(dims > 0, "bad dims");

  const auto options = CreateOptions();

  BENCHMARK("Gaussian Prior [AD]") {
    TestType x = TestType::Random(dims);
    ceres::Problem problem;
    problem.AddParameterBlock(x.data(), dims);  // Optimize the single variable 'x'
    ceres::CostFunction* cost_function;
    if constexpr (Dims == Eigen::Dynamic) {
      auto* cost = new ceres::DynamicAutoDiffCostFunction<MahalanobisCostFunctor<TestType>, 3>(
          new MahalanobisCostFunctor<TestType>(y));
      cost->AddParameterBlock(dims);
      cost->SetNumResiduals(dims);
      cost_function = cost;
    } else {
      cost_function = new ceres::AutoDiffCostFunction<MahalanobisCostFunctor<TestType>, Dims, Dims>(
          new MahalanobisCostFunctor<TestType>(y));
    }
    problem.AddResidualBlock(cost_function,
                             nullptr,           // No loss function.
                             x.data());         // The parameter block to which the cost function
    ceres::Solver::Summary summary;             // Summary of the optimization.
    ceres::Solve(options, &problem, &summary);  // Solve the problem!
    // std::cout << summary.FullReport() << "\n";  // Detailed report.
  };

  BENCHMARK("Gaussian Prior") {
    TestType x = TestType::Random(dims);
    ceres::Problem problem;
    problem.AddParameterBlock(x.data(), dims);  // Optimize the single variable 'x'
    ceres::CostFunction* cost_function;
    if constexpr (Dims == Eigen::Dynamic) {
      cost_function = new MahalanobisDynCostFunctor<TestType>(y);
    } else {
      cost_function = new MahalanobisFixedCostFunctor<TestType>(y);
    }
    problem.AddResidualBlock(cost_function,
                             nullptr,    // No loss function.
                             x.data());  // The parameter block to which the cost function applies.
    ceres::Solver::Summary summary;      // Summary of the optimization.
    ceres::Solve(options, &problem, &summary);  // Solve the problem!
    // std::cout << summary.FullReport() << "\n";  // Detailed report.
  };
}
