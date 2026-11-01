// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <tinyopt/tinyopt.h>

using Catch::Approx;
using namespace tinyopt;

TEST_CASE("tinyopt_optimizer_covariance_options_zero_fixed_variances") {
  Vec3 x = Vec3::Zero();
  Vec3 target(1.0, 2.0, 3.0);

  auto residuals = [&](const auto &value) { return (value - target).eval(); };

  lm::Optimizer<Mat3> optimizer;
  optimizer.fix({1});

  const auto summary = optimizer(x, residuals);
  REQUIRE(summary.Succeeded());

  // Default options: zero_fixed_variances = true
  CovarianceOptions cov_opts;
  cov_opts.zero_fixed_variances = true;
  const auto cov = optimizer.Covariance(cov_opts);
  REQUIRE(cov.has_value());

  // Fixed index 1 should have variance and cross-covariance = 0
  REQUIRE((*cov)(1, 1) == 0.0);
  REQUIRE((*cov)(0, 1) == 0.0);
  REQUIRE((*cov)(1, 0) == 0.0);
  REQUIRE((*cov)(1, 2) == 0.0);
  REQUIRE((*cov)(2, 1) == 0.0);

  // Free indices should have variance = 1.0
  REQUIRE((*cov)(0, 0) == Approx(1.0).margin(1e-10));
  REQUIRE((*cov)(2, 2) == Approx(1.0).margin(1e-10));

  // If zero_fixed_variances = false, the constrained identity entry gives 1.0
  cov_opts.zero_fixed_variances = false;
  const auto cov_raw = optimizer.Covariance(cov_opts);
  REQUIRE(cov_raw.has_value());
  REQUIRE((*cov_raw)(1, 1) == Approx(1.0).margin(1e-4));
}

TEST_CASE("tinyopt_optimizer_covariance_options_scale_residual_variance") {
  double x = 0.0;
  auto residuals = [](const auto &value) {
    using T = std::decay_t<decltype(value)>;
    Vector<T, 3> result;
    result << value - T(2.0), value - T(3.0), value - T(1.0);
    return result;
  };

  lm::Optimizer<Mat1> optimizer;
  const auto summary = optimizer(x, residuals);
  REQUIRE(summary.Succeeded());
  REQUIRE(summary.Converged());

  CovarianceOptions opts;
  opts.scale_by_residual_variance = true;
  const auto cov_scaled = optimizer.Covariance(opts);
  REQUIRE(cov_scaled.has_value());
  REQUIRE((*cov_scaled)(0, 0) == Approx(1.0 / 3.0).margin(1e-10));
}

TEST_CASE("tinyopt_summary_final_covariance_patterns") {
  Vec2 x = Vec2::Zero();
  auto residuals = [](const auto &value) { return (value - Vec2(1.0, 2.0)).eval(); };

  {
    Options options;
    options.exp.final_cov = true;
    options.exp.final_cov_pattern = CovariancePattern::Auto;
    const auto summary = Optimize(x, residuals, options);
    REQUIRE(summary.Succeeded());
    REQUIRE(std::holds_alternative<MatXf>(summary.final_cov));
    REQUIRE((std::get<MatXf>(summary.final_cov).template cast<double>() - Mat2::Identity()).norm() <
            1e-6);
  }
  {
    Options options;
    options.exp.final_cov = true;
    options.exp.final_cov_pattern = CovariancePattern::FullDense;
    const auto summary = Optimize(x, residuals, options);
    REQUIRE(summary.Succeeded());
    REQUIRE(std::holds_alternative<MatXf>(summary.final_cov));
    REQUIRE((std::get<MatXf>(summary.final_cov).template cast<double>() - Mat2::Identity()).norm() <
            1e-6);
  }
  {
    Options options;
    options.exp.final_cov = true;
    options.exp.final_cov_pattern = CovariancePattern::PreserveSparsity;
    const auto summary = Optimize(x, residuals, options);
    REQUIRE(summary.Succeeded());
    REQUIRE(std::holds_alternative<SparseMatf>(summary.final_cov));
    REQUIRE((MatX(std::get<SparseMatf>(summary.final_cov).template cast<double>().toDense()) -
             Mat2::Identity())
                .norm() < 1e-6);
  }
  {
    Options options;
    options.exp.final_cov = true;
    options.exp.final_cov_pattern = CovariancePattern::BlockDiagonal;
    const auto summary = Optimize(x, residuals, options);
    REQUIRE(summary.Succeeded());
    REQUIRE(std::holds_alternative<MatXf>(summary.final_cov));
    REQUIRE((std::get<MatXf>(summary.final_cov).template cast<double>() - Mat2::Identity()).norm() <
            1e-6);
  }
}
