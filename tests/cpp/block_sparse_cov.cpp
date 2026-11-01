// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <Eigen/Eigen>

#if CATCH2_VERSION == 2
#include <catch2/catch.hpp>
#else
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#endif

#include <tinyopt/tinyopt.h>

using Catch::Approx;
using namespace tinyopt;

TEST_CASE("tinyopt_optimizer2_block_sparse_options_and_fixed_dimensions") {
  using Hessian = BlockSparseMatrix<double, 2>;
  Vec6 x(4.0, -2.0, 0.5, -1.5, 5.0, 2.0);
  Vec6 target(1.0, 2.0, x[2], -0.5, 3.0, x[5]);

  Matrix<double, 6, 6> factor;
  factor << 2.0, 0.2, 0.1, 0.0, 0.0, 0.1, 0.0, 1.8, 0.2, 0.1, 0.0, 0.0, 0.1, 0.0, 2.2, 0.3, 0.1,
      0.0, 0.0, 0.1, 0.0, 1.7, 0.2, 0.1, 0.0, 0.0, 0.1, 0.0, 2.1, 0.2, 0.1, 0.0, 0.0, 0.1, 0.0, 1.9;
  const Matrix<double, 6, 6> dense_hessian =
      factor.transpose() * factor + Matrix<double, 6, 6>::Identity();
  Matrix<double, 12, 6> residual_factor;
  residual_factor.topRows<6>() = factor;
  residual_factor.bottomRows<6>().setIdentity();

  auto loss = [&](const Vec6& value, VecX& gradient, Hessian& hessian) {
    const Vec6 difference = value - target;
    const Vector<double, 12> residual = residual_factor * difference;
    gradient = dense_hessian * difference;
    for (Index row = 0; row < 3; ++row) {
      for (Index col = row; col < 3; ++col) {
        hessian.block<2, 2>(row, col) += dense_hessian.block<2, 2>(row * 2, col * 2);
      }
    }
    return 0.5 * residual.squaredNorm();
  };

  Options options(Options::Solver::GaussNewton);
  options.linear.method = LinearSolverMethod::LDLT;
  options.hessian.H_is_full = false;
  options.exp.final_cov = true;
  options.stop.max_iters = 20;
  gn::Optimizer<Hessian> optimizer(options);
  optimizer.fix({2, 5});

  const auto summary = optimizer.Optimize(x, loss);
  REQUIRE(summary.Succeeded());
  REQUIRE(summary.Converged());
  REQUIRE_FALSE(optimizer.useFixedPermutation());
  REQUIRE(x[2] == Approx(target[2]));
  REQUIRE(x[5] == Approx(target[5]));
  REQUIRE((x - target).norm() < 1e-6);
  REQUIRE(summary.has_final_cov());
  const MatX dense_covariance =
      std::get<BlockSparseMatX<float>>(summary.final_cov).toDense().template cast<double>();
  REQUIRE(dense_covariance.rows() == 6);
  REQUIRE(dense_covariance.cols() == 6);
  const auto* covariance = std::get_if<BlockSparseMatX<float>>(&summary.final_cov);
  REQUIRE(covariance != nullptr);
  REQUIRE(covariance->block_sizes() == std::vector<int>{2});
  REQUIRE(covariance->block_counts() == std::vector<Index>{3});
  MatX constrained_hessian = dense_hessian;
  for (const Index index : {Index(2), Index(5)}) {
    constrained_hessian.row(index).setZero();
    constrained_hessian.col(index).setZero();
    constrained_hessian(index, index) = 1;
  }
  REQUIRE((covariance->toDense().template cast<double>() - constrained_hessian.inverse()).norm() <
          1e-6);

  const MatX expected_covariance = constrained_hessian.inverse();
  const auto run_with_pattern = [&](CovariancePattern pattern) {
    Vec6 pattern_x(4.0, -2.0, 0.5, -1.5, 5.0, 2.0);
    Options pattern_options = options;
    pattern_options.exp.final_cov_pattern = pattern;
    gn::Optimizer<Hessian> pattern_optimizer(pattern_options);
    pattern_optimizer.fix({2, 5});
    const auto pattern_summary = pattern_optimizer.Optimize(pattern_x, loss);
    REQUIRE(pattern_summary.Succeeded());
    return pattern_summary;
  };

  const auto full_dense = run_with_pattern(CovariancePattern::FullDense);
  REQUIRE(std::holds_alternative<MatXf>(full_dense.final_cov));
  REQUIRE(
      (std::get<MatXf>(full_dense.final_cov).template cast<double>() - expected_covariance).norm() <
      1e-6);

  const auto block_diagonal = run_with_pattern(CovariancePattern::BlockDiagonal);
  REQUIRE(std::holds_alternative<BlockSparseMatX<float>>(block_diagonal.final_cov));
  const MatX block_diagonal_dense =
      std::get<BlockSparseMatX<float>>(block_diagonal.final_cov).toDense().template cast<double>();
  for (Index row = 0; row < 3; ++row) {
    REQUIRE((block_diagonal_dense.block<2, 2>(row * 2, row * 2) -
             expected_covariance.block<2, 2>(row * 2, row * 2))
                .norm() < 1e-6);
  }
  REQUIRE(block_diagonal_dense.block<2, 2>(0, 2).norm() < 1e-7);
  REQUIRE(block_diagonal_dense.block<2, 2>(0, 4).norm() < 1e-7);
  REQUIRE(block_diagonal_dense.block<2, 2>(2, 4).norm() < 1e-7);

  const auto preserve_sparsity = run_with_pattern(CovariancePattern::PreserveSparsity);
  REQUIRE(std::holds_alternative<SparseMatf>(preserve_sparsity.final_cov));
  REQUIRE(
      (MatX(std::get<SparseMatf>(preserve_sparsity.final_cov).template cast<double>().toDense()) -
       expected_covariance)
          .norm() < 1e-6);
}
