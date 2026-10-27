// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <vector>

#include <Eigen/Eigen>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <tinyopt/tinyopt.h>

#include <tinyopt/math/block_sparse_ldlt_dynamic.h>
#include <tinyopt/math/solvers/block_sparse.h>
#include <tinyopt/optimizers/lm.h>

using Catch::Approx;

using namespace tinyopt;

namespace {

BlockSparseMatX<double> MakeEmptyDynamic(Index dims, std::vector<int> sizes,
                                         std::vector<Index> counts) {
  SparseMatrix<double> empty(dims, dims);
  return BlockSparseMatX<double>(std::move(empty), std::move(sizes), std::move(counts));
}

MatX AssembleTridiagonal(const std::vector<MatX>& diagonal, const std::vector<MatX>& off_diagonal) {
  Index dims = 0;
  for (const auto& block : diagonal) dims += block.rows();
  MatX dense = MatX::Zero(dims, dims);
  Index offset = 0;
  for (size_t i = 0; i < diagonal.size(); ++i) {
    const Index rows = diagonal[i].rows();
    dense.block(offset, offset, rows, rows) = diagonal[i];
    if (i + 1 < diagonal.size()) {
      dense.block(offset, offset + rows, rows, off_diagonal[i].cols()) = off_diagonal[i];
      dense.block(offset + rows, offset, off_diagonal[i].cols(), rows) =
          off_diagonal[i].transpose();
    }
    offset += rows;
  }
  return dense;
}

}  // namespace

TEST_CASE("block_sparse_dynamic_ldlt_matches_dense") {
  const Mat2 d0{{4.0, 1.0}, {1.0, 3.0}};
  const Mat2 d1{{5.0, 0.5}, {0.5, 4.0}};
  const Mat2 d2{{3.0, 0.2}, {0.2, 2.0}};
  const Mat2 u0{{0.5, 0.1}, {0.2, 0.3}};
  const Mat2 u1{{0.1, 0.0}, {0.3, 0.2}};

  auto H = MakeEmptyDynamic(6, {2}, {3});
  H.setBlock(0, 0, d0.data());
  H.setBlock(1, 1, d1.data());
  H.setBlock(2, 2, d2.data());
  H.setBlock(0, 1, u0.data());
  H.setBlock(1, 2, u1.data());

  const MatX dense = AssembleTridiagonal({d0, d1, d2}, {u0, u1});
  REQUIRE((H.toDense() - dense).norm() < 1e-12);

  RuntimeBlockSparseLDLT<BlockSparseMatX<double>> ldlt;
  ldlt.analyzePattern(H);
  REQUIRE(ldlt.isAnalyzed());
  ldlt.compute(H);
  REQUIRE(ldlt.isComputed());
  REQUIRE(ldlt.info() == Eigen::Success);

  const VecX rhs = VecX::LinSpaced(6, 1.0, 6.0);
  const VecX solution = ldlt.solve(rhs);
  REQUIRE((dense * solution - rhs).norm() < 1e-10);

  const MatX identity = MatX::Identity(6, 6);
  const MatX inverse = ldlt.solve(identity);
  REQUIRE((dense * inverse - identity).norm() < 1e-10);

  BlockSparseLinearSolvers<BlockSparseMatX<double>> solver;
  const auto solved = solver.Solve(H, rhs);
  REQUIRE(solved.has_value());
  REQUIRE((dense * *solved - rhs).norm() < 1e-10);

  const auto covariance = solver.Inverse(H);
  REQUIRE(covariance.has_value());
  REQUIRE((covariance->toDense() - inverse).norm() < 1e-10);
}

TEST_CASE("block_sparse_dynamic_heterogeneous_ldlt_matches_dense") {
  const Mat2 d0{{4.0, 0.5}, {0.5, 3.0}};
  const Mat3 d1{{6.0, 0.2, 0.1}, {0.2, 5.0, 0.3}, {0.1, 0.3, 4.0}};
  const Matrix<double, 2, 3> u0{{0.2, 0.1, 0.0}, {0.3, 0.0, 0.4}};

  auto H = MakeEmptyDynamic(5, {2, 3}, {1, 1});
  H.setBlock(0, 0, d0.data());
  H.setBlock(1, 1, d1.data());
  H.setBlock(0, 1, u0.data());

  MatX dense = MatX::Zero(5, 5);
  dense.block<2, 2>(0, 0) = d0;
  dense.block<3, 3>(2, 2) = d1;
  dense.block<2, 3>(0, 2) = u0;
  dense.block<3, 2>(2, 0) = u0.transpose();
  REQUIRE((H.toDense() - dense).norm() < 1e-12);

  RuntimeBlockSparseLDLT<BlockSparseMatX<double>> ldlt;
  ldlt.analyzePattern(H);
  ldlt.compute(H);
  REQUIRE(ldlt.isComputed());
  const VecX rhs = VecX::Ones(5);
  const VecX solution = ldlt.solve(rhs);
  REQUIRE((dense * solution - rhs).norm() < 1e-10);

  const auto inverse = Inverse(H);
  REQUIRE(inverse.has_value());
  REQUIRE((MatX(*inverse) * dense - MatX::Identity(5, 5)).norm() < 1e-10);
}

TEST_CASE("block_sparse_dynamic_optimizer_writes_blocks") {
  const Vec6 target(1.0, -2.0, 0.5, 3.0, -1.0, 2.0);
  const Mat2 d0{{4.0, 1.0}, {1.0, 3.0}};
  const Mat2 d1{{5.0, 0.5}, {0.5, 4.0}};
  const Mat2 d2{{3.0, 0.2}, {0.2, 2.0}};
  const Mat2 u0{{0.5, 0.1}, {0.2, 0.3}};
  const Mat2 u1{{0.1, 0.0}, {0.3, 0.2}};
  const MatX hessian = AssembleTridiagonal({d0, d1, d2}, {u0, u1});

  auto loss = [&](const VecX& value, VecX& gradient, BlockSparseMatX<double>& H) {
    const VecX difference = value - target;
    gradient.noalias() = hessian * difference;
    H.setBlock(0, 0, d0.data());
    H.setBlock(1, 1, d1.data());
    H.setBlock(2, 2, d2.data());
    H.setBlock(0, 1, u0.data());
    H.setBlock(1, 2, u1.data());
    return 0.5 * difference.dot(hessian * difference);
  };

  VecX x = VecX::Zero(6);
  Options options(Options::Solver::LevenbergMarquardt);
  options.lm.jacobi_scaling = true;
  options.lm.damping_init = 1e-4;
  options.stop.max_iters = 20;
  options.exp.final_cov = true;

  lm::Optimizer<BlockSparseMatX<double>> optimizer(options);
  const auto prototype = MakeEmptyDynamic(6, {2}, {3});
  optimizer.InitWith(VecX::Zero(6), prototype);
  const auto summary = optimizer.Optimize(x, loss);

  REQUIRE(summary.Succeeded());
  REQUIRE(summary.Converged());
  REQUIRE((x - VecX(target)).norm() < 1e-6);
  REQUIRE(summary.has_final_cov());
  REQUIRE(std::holds_alternative<BlockSparseMatX<float>>(summary.final_cov));
  const auto& covariance = std::get<BlockSparseMatX<float>>(summary.final_cov);
  REQUIRE(covariance.block_sizes() == std::vector<int>{2});
  REQUIRE(covariance.block_counts() == std::vector<Index>{3});
  const MatX expected = hessian.inverse();
  REQUIRE((covariance.toDense().template cast<double>() - expected).norm() < 1e-4);
}

TEST_CASE("block_sparse_dynamic_static_conversion_preserves_layout") {
  BlockSparseMatrix<double, 2, 3> static_matrix(1, 1);
  static_matrix.block<2, 2>(0, 0) = Mat2::Identity();
  static_matrix.block<3, 3>(0, 0) = Mat3::Identity();
  const auto dynamic = static_matrix.toBlockSparseMatX();
  REQUIRE(dynamic.block_sizes() == std::vector<int>{2, 3});
  REQUIRE(dynamic.block_counts() == std::vector<Index>{1, 1});
  REQUIRE((dynamic.toDense() - static_matrix.toDense()).norm() < 1e-12);
}
