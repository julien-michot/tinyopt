// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <tinyopt/tinyopt.h>

using Catch::Approx;
using namespace tinyopt;

TEST_CASE("tinyopt_block_tridiag_covariance_takahashi") {
  constexpr int B = 3;
  constexpr Index N = 4;
  BlockTridiagMat<B> T(N);

  for (Index i = 0; i < N; ++i) {
    Matrix<double, B, B> D = Matrix<double, B, B>::Random();
    T.diagonal(i) = D * D.transpose() + Matrix<double, B, B>::Identity() * 5.0;
  }
  for (Index i = 0; i < N - 1; ++i) {
    T.off_diagonal(i) = Matrix<double, B, B>::Random();
  }

  const auto T_inv = Inverse(T);
  REQUIRE(T_inv.has_value());

  const Eigen::MatrixXd dense_inv = T.toDense().inverse();
  for (Index i = 0; i < N; ++i) {
    REQUIRE((T_inv->diagonal(i) - dense_inv.block<B, B>(i * B, i * B)).norm() < 1e-10);
  }
  for (Index i = 0; i < N - 1; ++i) {
    REQUIRE((T_inv->off_diagonal(i) - dense_inv.block<B, B>(i * B, (i + 1) * B)).norm() < 1e-10);
  }

  // Test BlockCovariance on BlockTridiagLinearSolvers
  BlockTridiagLinearSolvers<BlockTridiagMat<B>> solver;
  const auto cov_00 = solver.BlockCovariance(T, 0, 0);
  REQUIRE(cov_00.has_value());
  REQUIRE((*cov_00 - dense_inv.block<B, B>(0, 0)).norm() < 1e-10);

  const auto cov_02 = solver.BlockCovariance(T, 0, 2);
  REQUIRE(cov_02.has_value());
  REQUIRE((*cov_02 - dense_inv.block<B, B>(0, 2 * B)).norm() < 1e-10);
}

TEST_CASE("tinyopt_optimizer_block_covariance_and_variances") {
  using Vec6 = Vector<double, 6>;
  Vec6 x = Vec6::Zero();
  const Vec6 target = Vec6::Constant(2.0);

  auto residuals = [&](const auto &value) { return (value - target).eval(); };

  lm::Optimizer<Mat6> optimizer;
  const auto summary = optimizer(x, residuals);
  REQUIRE(summary.Succeeded());

  const auto std_devs = optimizer.BlockStandardDeviations();
  for (int i = 0; i < 6; ++i) {
    REQUIRE(std_devs[i] == Approx(1.0).margin(1e-5));
  }

  const auto diag_blocks = optimizer.BlockDiagonalCovariance();
  REQUIRE(diag_blocks.size() == 6);
  for (size_t i = 0; i < 6; ++i) {
    REQUIRE(diag_blocks[i](0, 0) == Approx(1.0).margin(1e-5));
  }

  const auto block_00 = optimizer.BlockCovariance(0, 0);
  REQUIRE(block_00.has_value());
  REQUIRE((*block_00)(0, 0) == Approx(1.0).margin(1e-5));
}

namespace {

struct SchurCovTestParams {
  using Scalar = double;
  static constexpr Index Dims = 5;

  Vector<double, Dims> values = Vector<double, Dims>::Zero();
  std::pair<Index, Index> reduced() const { return {0, 2}; }

  SchurCovTestParams &operator+=(const Vector<double, Dims> &delta) {
    values += delta;
    return *this;
  }
};

}  // namespace

TEST_CASE("tinyopt_optimizer_schur_covariance") {
  using Hessian = BlockSparseMatrix<double, 2, 3>;
  SchurCovTestParams parameters;
  parameters.values << 2.0, -1.0, 1.0, 0.5, -0.5;
  const Vector<double, 5> target = Vector<double, 5>::Zero();

  Matrix<double, 2, 2> reduced_H = 3.0 * Matrix<double, 2, 2>::Identity();
  Matrix<double, 3, 3> eliminated_H = 4.0 * Matrix<double, 3, 3>::Identity();
  Matrix<double, 2, 3> coupling = Matrix<double, 2, 3>::Zero();
  coupling(0, 0) = 0.5;
  coupling(1, 1) = -0.3;

  Matrix<double, 5, 5> dense_H = Matrix<double, 5, 5>::Zero();
  dense_H.topLeftCorner<2, 2>() = reduced_H;
  dense_H.topRightCorner<2, 3>() = coupling;
  dense_H.bottomLeftCorner<3, 2>() = coupling.transpose();
  dense_H.bottomRightCorner<3, 3>() = eliminated_H;

  Eigen::LLT<Matrix<double, 5, 5>> llt(dense_H);
  REQUIRE(llt.info() == Eigen::Success);
  const Matrix<double, 5, 5> residual_factor = llt.matrixL().toDenseMatrix().transpose();

  auto loss = [&](const SchurCovTestParams &value, Vector<double, Dynamic> &grad, Hessian &H) {
    const Vector<double, 5> diff = value.values - target;
    const Vector<double, 5> res = residual_factor * diff;
    grad = dense_H * diff;
    H.block<2, 2>(0, 0) = reduced_H;
    H.block<2, 3>(0, 0) = coupling;
    H.block<3, 3>(0, 0) = eliminated_H;
    return 0.5 * res.squaredNorm();
  };

  Options options(Options::Solver::LevenbergMarquardt);
  options.lm.damping_init = 1e-4;
  lm::Optimizer<Hessian> optimizer(options);
  const auto summary = optimizer(parameters, loss);
  REQUIRE(summary.Succeeded());
  REQUIRE(summary.Converged());

  const auto schur_cov = optimizer.SchurCovariance();

  const Matrix<double, 5, 5> dense_inv = dense_H.inverse();

  // 1. Check reduced (camera) covariance
  const Mat2 cov_reduced = schur_cov.reduced();
  REQUIRE((cov_reduced - dense_inv.topLeftCorner<2, 2>()).norm() < 1e-10);

  // 2. Check eliminated (point) marginal block
  const Matrix<double, 3, 3> cov_pt0 = schur_cov.template eliminated_block<3>(0);
  REQUIRE((cov_pt0 - dense_inv.bottomRightCorner<3, 3>()).norm() < 1e-10);

  // 3. Check eliminated blocks vector
  const auto eliminated_blocks = schur_cov.template eliminated_blocks<3>();
  REQUIRE(eliminated_blocks.size() == 1);
  REQUIRE((eliminated_blocks[0] - dense_inv.bottomRightCorner<3, 3>()).norm() < 1e-10);
}
