// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <vector>

#include <Eigen/Eigen>

#if CATCH2_VERSION == 2
#include <catch2/catch.hpp>
#else
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#endif

#include <tinyopt/tinyopt.h>

#include <tinyopt/diff/gradient_check.h>
#include <tinyopt/diff/num_diff.h>
#include <tinyopt/optimize.h>
#include <tinyopt/optimizers/optimizer.h>

using Catch::Approx;

using namespace tinyopt;
using namespace tinyopt::diff;

struct BlockSchurParams {
  using Scalar = double;
  static constexpr Index Dims = 5;

  Vector<double, Dims> values = Vector<double, Dims>::Zero();
  std::vector<Index> locked_indices;

  std::pair<Index, Index> reduced() const { return {0, 2}; }
  const std::vector<Index>& locked() const { return locked_indices; }
  BlockSchurParams& operator+=(const Vector<double, Dims>& delta) {
    values += delta;
    return *this;
  }
};

TEST_CASE("tinyopt_optimize_block_sparse_hessian") {
  Vec4 x(8.0, -6.0, 4.0, -5.0);
  using Hessian = BlockSparseMatrix<double, 2>;
  const Matrix<double, 2, 2> first_factor{{2.0, 0.3}, {0.3, 4.0}};
  const Matrix<double, 2, 2> second_factor{{3.0, 0.2}, {0.2, 5.0}};
  const Vec4 target(2.0, -1.0, 0.5, 3.0);

  Matrix<double, 4, 4> residual_factor = Matrix<double, 4, 4>::Zero();
  residual_factor.topLeftCorner<2, 2>() = first_factor;
  residual_factor.bottomRightCorner<2, 2>() = second_factor;
  const Matrix<double, 4, 4> dense_hessian = residual_factor.transpose() * residual_factor;

  auto gradient_only = [&](const Vec4& value, auto& gradient) {
    const Vec4 difference = value - target;
    const Vec4 residual = residual_factor * difference;
    if constexpr (!traits::is_nullptr_v<decltype(gradient)>) gradient = dense_hessian * difference;
    return 0.5 * residual.squaredNorm();
  };
  REQUIRE(CheckGradient(x, gradient_only, 1e-5, Method::kCentral, false));

  auto loss = [&](const Vec4& value, Vector<double, Dynamic>& gradient, Hessian& hessian) {
    const Vec4 difference = value - target;
    const Vec4 residual = residual_factor * difference;
    gradient = dense_hessian * difference;
    hessian.block<2, 2>(0, 0) = dense_hessian.topLeftCorner<2, 2>();
    hessian.block<2, 2>(1, 1) = dense_hessian.bottomRightCorner<2, 2>();
    return 0.5 * residual.squaredNorm();
  };

  Options options(Options::Solver::LevenbergMarquardt);
  options.lm.jacobi_scaling = true;
  options.lm.damping_init = 1e-3;
  options.stop.max_iters = 20;
  const auto summary = Optimize(x, loss, options);

  REQUIRE(summary.Succeeded());
  REQUIRE(summary.Converged());
  REQUIRE((x - target).norm() < 1e-6);
}

TEST_CASE("tinyopt_optimizer2_block_sparse_options_and_locked_dimensions") {
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

  auto gradient_only = [&](const Vec6& value, auto& gradient) {
    const Vec6 difference = value - target;
    const Vector<double, 12> residual = residual_factor * difference;
    if constexpr (!traits::is_nullptr_v<decltype(gradient)>) gradient = dense_hessian * difference;
    return 0.5 * residual.squaredNorm();
  };
  REQUIRE(CheckGradient(x, gradient_only, 1e-5, Method::kCentral, false));

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
  optimizer.lock({2, 5});

  const auto summary = optimizer.Optimize(x, loss);
  REQUIRE(summary.Succeeded());
  REQUIRE(summary.Converged());
  REQUIRE_FALSE(optimizer.useLockedPermutation());
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
    pattern_optimizer.lock({2, 5});
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

TEST_CASE("tinyopt_optimize_block_sparse_schur_hessian") {
  BlockSchurParams parameters;
  parameters.values << 5.0, -3.0, 4.0, 7.0, -2.0;
  parameters.locked_indices = {1, 4};
  const Vector<double, 5> target{1.0, -3.0, -1.0, 0.5, -2.0};
  using Hessian = BlockSparseMatrix<double, 2, 3>;
  Matrix<double, 2, 2> reduced_hessian = 2.0 * Matrix<double, 2, 2>::Identity();
  Matrix<double, 3, 3> eliminated_hessian = 3.0 * Matrix<double, 3, 3>::Identity();
  Matrix<double, 2, 3> coupling = Matrix<double, 2, 3>::Zero();
  coupling(0, 0) = 0.25;
  coupling(1, 1) = -0.2;
  Matrix<double, 5, 5> dense_hessian = Matrix<double, 5, 5>::Zero();
  dense_hessian.topLeftCorner<2, 2>() = reduced_hessian;
  dense_hessian.topRightCorner<2, 3>() = coupling;
  dense_hessian.bottomLeftCorner<3, 2>() = coupling.transpose();
  dense_hessian.bottomRightCorner<3, 3>() = eliminated_hessian;
  Eigen::LLT<Matrix<double, 5, 5>> hessian_factorization(dense_hessian);
  REQUIRE(hessian_factorization.info() == Eigen::Success);
  const Matrix<double, 5, 5> residual_factor =
      hessian_factorization.matrixL().toDenseMatrix().transpose();

  auto loss = [&](const BlockSchurParams& value, Vector<double, Dynamic>& gradient,
                  Hessian& hessian) {
    const Vector<double, 5> difference = value.values - target;
    const Vector<double, 5> residual = residual_factor * difference;
    gradient = dense_hessian * difference;
    hessian.block<2, 2>(0, 0) = reduced_hessian;
    hessian.block<2, 3>(0, 0) = coupling;
    hessian.block<3, 3>(0, 0) = eliminated_hessian;
    return 0.5 * residual.squaredNorm();
  };

  Options options(Options::Solver::LevenbergMarquardt);
  options.lm.jacobi_scaling = true;
  options.lm.damping_init = 1e-3;
  options.exp.final_cov = true;
  lm::Optimizer<Hessian> optimizer(options);
  const auto summary = optimizer(parameters, loss);
  REQUIRE(summary.Succeeded());
  REQUIRE(summary.Converged());
  REQUIRE(parameters.values[1] == Approx(-3.0));
  REQUIRE(parameters.values[4] == Approx(-2.0));
  REQUIRE((parameters.values - target).norm() < 1e-6);
  const auto* covariance = std::get_if<BlockSparseMatX<float>>(&summary.final_cov);
  REQUIRE(covariance != nullptr);
  REQUIRE(covariance->block_sizes() == std::vector<int>{2, 3});
  REQUIRE(covariance->block_counts() == std::vector<Index>{1, 1});
  REQUIRE((covariance->toDense().template cast<double>() - optimizer.Hessian().toDense().inverse())
              .norm() < 1e-6);
}
