// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <tinyopt/math/block_tridiag_matrix.h>
#include <tinyopt/optimize.h>
#include <tinyopt/types.h>

using namespace tinyopt;

TEST_CASE("Optimizer with BlockTridiagMat Hessian", "[tridiagonal][optimizer]") {
  // State: 4 nodes of dimension 2 (Vec8)
  using Hessian = BlockTridiagMat<2>;
  using Vec8 = Vector<double, 8>;
  Vec8 x;
  x << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0;
  const Vec8 x0 = x;

  const Vec8 target = Vec8::Constant(2.0);

  auto loss = [&](const Vec8& val, Vector<double, Dynamic>& grad, Hessian& H) {
    H.setZero();
    grad.setZero();
    double total_cost = 0.0;

    // Node target priors: 0.5 * ||x_i - target_i||^2
    for (Index i = 0; i < 4; ++i) {
      Vec2 diff = val.segment<2>(i * 2) - target.segment<2>(i * 2);
      total_cost += 0.5 * diff.squaredNorm();
      grad.segment<2>(i * 2) += diff;
      H.diagonal(i).noalias() += Matrix<double, 2, 2>::Identity();
    }

    // Edge smoothness coupling: 0.5 * w * ||x_{i+1} - x_i||^2
    const double w = 2.0;
    for (Index i = 0; i < 3; ++i) {
      Vec2 diff = val.segment<2>((i + 1) * 2) - val.segment<2>(i * 2);
      total_cost += 0.5 * w * diff.squaredNorm();
      grad.segment<2>(i * 2) -= w * diff;
      grad.segment<2>((i + 1) * 2) += w * diff;

      H.diagonal(i).noalias() += w * Matrix<double, 2, 2>::Identity();
      H.diagonal(i + 1).noalias() += w * Matrix<double, 2, 2>::Identity();
      H.off_diagonal(i).noalias() -= w * Matrix<double, 2, 2>::Identity();
    }

    return total_cost;
  };

  Options options(Options::Solver::LevenbergMarquardt);
  options.lm.damping_init = 1e-4;
  options.stop.max_iters = 20;

  const auto summary = Optimize(x, loss, options);
  REQUIRE(summary.Succeeded());
  REQUIRE(summary.Converged());
  REQUIRE((x - target).norm() < 1e-6);

  x = x0;
  Options cov_options(Options::Solver::LevenbergMarquardt);
  cov_options.lm.damping_init = 1e-4;
  cov_options.stop.max_iters = 20;
  cov_options.exp.final_cov = true;
  cov_options.exp.final_cov_pattern = CovariancePattern::BlockDiagonal;
  const auto cov_summary = Optimize(x, loss, cov_options);
  REQUIRE(cov_summary.has_final_cov());
  REQUIRE(std::holds_alternative<BlockTridiagMatX<float>>(cov_summary.final_cov));

  Hessian expected_hessian(4);
  for (Index i = 0; i < 4; ++i) {
    const double weight = (i == 0 || i == 3) ? 3.0 : 5.0;
    expected_hessian.diagonal(i) = weight * Mat2::Identity();
  }
  for (Index i = 0; i < 3; ++i) {
    expected_hessian.off_diagonal(i) = -2.0 * Mat2::Identity();
  }
  const auto full_covariance = Inverse(expected_hessian);
  REQUIRE(full_covariance.has_value());
  const MatX expected_dense = full_covariance->toDense();
  const MatX block_diagonal =
      std::get<BlockTridiagMatX<float>>(cov_summary.final_cov).toDense().template cast<double>();
  for (Index i = 0; i < 4; ++i) {
    REQUIRE((block_diagonal.block<2, 2>(i * 2, i * 2) - expected_dense.block<2, 2>(i * 2, i * 2))
                .norm() < 1e-6);
  }
  REQUIRE(block_diagonal.block<2, 2>(0, 2).norm() < 1e-7);
  REQUIRE(block_diagonal.block<2, 2>(2, 6).norm() < 1e-7);
}
