// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <limits>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <tinyopt/math/block_sparse_ldlt.h>
#include <tinyopt/math/block_sparse_matrix.h>
#include <tinyopt/types.h>

using namespace tinyopt;

TEST_CASE("block_sparse_ldlt basic solve and ordering", "[solver][ldlt]") {
  // 2 cameras (size 6), 3 points (size 3) -> 21 total dimension
  BlockSparseMat<6, 3> H(2, 3);

  // Fill positive-definite diagonal blocks
  for (Index i = 0; i < 2; ++i) {
    Matrix<double, 6, 6> B = Matrix<double, 6, 6>::Random();
    H.diagonal<6>(i) = B * B.transpose() + Matrix<double, 6, 6>::Identity() * 10.0;
  }
  for (Index j = 0; j < 3; ++j) {
    Matrix<double, 3, 3> C = Matrix<double, 3, 3>::Random();
    H.diagonal<3>(j) = C * C.transpose() + Matrix<double, 3, 3>::Identity() * 5.0;
  }

  // Camera-camera off-diagonal
  H.block<6, 6>(0, 1) = Matrix<double, 6, 6>::Random() * 0.1;

  // Camera-point observations
  H.block<6, 3>(0, 0) = Matrix<double, 6, 3>::Random();
  H.block<6, 3>(0, 1) = Matrix<double, 6, 3>::Random();
  H.block<6, 3>(1, 1) = Matrix<double, 6, 3>::Random();
  H.block<6, 3>(1, 2) = Matrix<double, 6, 3>::Random();

  const VecX x_expected = VecX::Random(21);
  const VecX b = H * x_expected;

  SECTION("BlockGuidedOrdering eliminates points first") {
    BlockSparseLDLT<decltype(H), BlockGuidedOrdering<1>> ldlt;
    ldlt.analyzePattern(H);
    REQUIRE(ldlt.isAnalyzed());
    REQUIRE(ldlt.isBlockDiagonalEliminated());
    REQUIRE(ldlt.numReduced() == 12);
    REQUIRE(ldlt.numEliminated() == 9);

    ldlt.compute(H);
    REQUIRE(ldlt.isComputed());
    REQUIRE(ldlt.info() == Eigen::Success);

    const VecX x_sol = ldlt.solve(b);
    REQUIRE((x_sol - x_expected).norm() < 1e-10);

    const auto opt_sol = ldlt.Solve(b);
    REQUIRE(opt_sol.has_value());
    REQUIRE((*opt_sol - x_expected).norm() < 1e-10);
  }

  SECTION("BlockAMDOrdering automatically detects block diagonal elimination") {
    BlockSparseLDLT<decltype(H), BlockAMDOrdering> ldlt;
    ldlt.analyzePattern(H, 12);
    REQUIRE(ldlt.isAnalyzed());
    REQUIRE(ldlt.isBlockDiagonalEliminated());

    ldlt.compute(H);
    REQUIRE(ldlt.isComputed());
    REQUIRE(ldlt.info() == Eigen::Success);

    const VecX x_sol = ldlt.solve(b);
    REQUIRE((x_sol - x_expected).norm() < 1e-10);
  }

  SECTION("BlockNaturalOrdering with natural block partition") {
    BlockSparseLDLT<decltype(H), BlockNaturalOrdering> ldlt(H, 12);
    REQUIRE(ldlt.isAnalyzed());
    REQUIRE(ldlt.isComputed());
    REQUIRE(ldlt.info() == Eigen::Success);

    const VecX x_sol = ldlt.solve(b);
    REQUIRE((x_sol - x_expected).norm() < 1e-10);
  }

  SECTION("full direct matrix factorization without Schur partition") {
    BlockSparseLDLT<decltype(H), BlockAMDOrdering> ldlt;
    ldlt.analyzePattern(H, 0);  // 0 means no Schur partitioning
    REQUIRE(ldlt.isAnalyzed());
    REQUIRE(ldlt.numReduced() == 0);

    ldlt.compute(H);
    REQUIRE(ldlt.isComputed());
    REQUIRE(ldlt.info() == Eigen::Success);

    const VecX x_sol = ldlt.solve(b);
    REQUIRE((x_sol - x_expected).norm() < 1e-10);
  }
}

TEST_CASE("block_sparse_ldlt matrix solve and inverse recovery", "[solver][ldlt]") {
  BlockSparseMat<6, 3> H(2, 3);
  for (Index i = 0; i < 2; ++i) {
    Matrix<double, 6, 6> B = Matrix<double, 6, 6>::Random();
    H.diagonal<6>(i) = B * B.transpose() + Matrix<double, 6, 6>::Identity() * 8.0;
  }
  for (Index j = 0; j < 3; ++j) {
    Matrix<double, 3, 3> C = Matrix<double, 3, 3>::Random();
    H.diagonal<3>(j) = C * C.transpose() + Matrix<double, 3, 3>::Identity() * 6.0;
  }
  H.block<6, 3>(0, 0) = Matrix<double, 6, 3>::Random();
  H.block<6, 3>(1, 1) = Matrix<double, 6, 3>::Random();
  H.block<6, 3>(1, 2) = Matrix<double, 6, 3>::Random();

  BlockSparseLDLT<decltype(H), BlockGuidedOrdering<1>> ldlt(H);
  REQUIRE(ldlt.isComputed());

  SECTION("recover full inverse using identity matrix") {
    const Matrix<double, Dynamic, Dynamic> I = Matrix<double, Dynamic, Dynamic>::Identity(21, 21);
    const Matrix<double, Dynamic, Dynamic> H_inv = ldlt.solve(I);
    REQUIRE(H_inv.allFinite());

    // Verify H * H_inv = I
    const Matrix<double, Dynamic, Dynamic> H_dense = H.toDense();
    const Matrix<double, Dynamic, Dynamic> identity_check = H_dense * H_inv;
    REQUIRE((identity_check - I).norm() < 1e-10);
  }

  SECTION("recover partial inverse for camera parameters") {
    // Select first 6 columns (camera 0)
    const Matrix<double, Dynamic, Dynamic> I_cam0 =
        Matrix<double, Dynamic, Dynamic>::Identity(21, 21).leftCols(6);
    const Matrix<double, Dynamic, Dynamic> H_inv_cam0 = ldlt.solve(I_cam0);
    REQUIRE(H_inv_cam0.rows() == 21);
    REQUIRE(H_inv_cam0.cols() == 6);

    const Matrix<double, Dynamic, Dynamic> H_dense = H.toDense();
    const Matrix<double, Dynamic, Dynamic> check = H_dense * H_inv_cam0;
    REQUIRE((check - I_cam0).norm() < 1e-10);
  }
}

TEST_CASE("block_sparse_ldlt coupled eliminated variables (distance prior)", "[solver][ldlt]") {
  // In this test, points 0 and 1 are connected by an off-diagonal prior block!
  // Traditional Schur solvers reject this, but BlockSparseLDLT detects and solves it seamlessly.
  BlockSparseMat<6, 3> H(2, 3);
  for (Index i = 0; i < 2; ++i) {
    Matrix<double, 6, 6> B = Matrix<double, 6, 6>::Random();
    H.diagonal<6>(i) = B * B.transpose() + Matrix<double, 6, 6>::Identity() * 10.0;
  }
  for (Index j = 0; j < 3; ++j) {
    Matrix<double, 3, 3> C = Matrix<double, 3, 3>::Random();
    H.diagonal<3>(j) = C * C.transpose() + Matrix<double, 3, 3>::Identity() * 5.0;
  }
  H.block<6, 3>(0, 0) = Matrix<double, 6, 3>::Random();
  H.block<6, 3>(1, 1) = Matrix<double, 6, 3>::Random();

  // Cross-coupling between point 0 and point 1
  H.block<3, 3>(0, 1) = Matrix<double, 3, 3>::Constant(0.2);

  const VecX x_expected = VecX::Random(21);
  const VecX b = H * x_expected;

  BlockSparseLDLT<decltype(H), BlockGuidedOrdering<1>> ldlt;
  ldlt.analyzePattern(H);
  REQUIRE(ldlt.isAnalyzed());
  // Verify that coupled eliminated variables are properly detected
  REQUIRE_FALSE(ldlt.isBlockDiagonalEliminated());

  ldlt.compute(H);
  REQUIRE(ldlt.isComputed());
  REQUIRE(ldlt.info() == Eigen::Success);

  const VecX x_sol = ldlt.solve(b);
  REQUIRE((x_sol - x_expected).norm() < 1e-10);

  // Also test matrix solve on coupled eliminated system
  const Matrix<double, Dynamic, Dynamic> I = Matrix<double, Dynamic, Dynamic>::Identity(21, 21);
  const Matrix<double, Dynamic, Dynamic> H_inv = ldlt.solve(I);
  REQUIRE((H.toDense() * H_inv - I).norm() < 1e-10);
}

TEST_CASE("block_sparse_ldlt BA with non-consecutive points sharing distance prior",
          "[solver][ldlt]") {
  // Bundle adjustment scenario with 4 cameras (dim 6 each) and 8 points (dim 3 each) = 48 total dim
  BlockSparseMat<6, 3> H(4, 8);

  // Positive-definite diagonal camera blocks
  for (Index i = 0; i < 4; ++i) {
    Matrix<double, 6, 6> B = Matrix<double, 6, 6>::Random();
    H.diagonal<6>(i) = B * B.transpose() + Matrix<double, 6, 6>::Identity() * 15.0;
  }
  // Positive-definite diagonal point blocks
  for (Index j = 0; j < 8; ++j) {
    Matrix<double, 3, 3> C = Matrix<double, 3, 3>::Random();
    H.diagonal<3>(j) = C * C.transpose() + Matrix<double, 3, 3>::Identity() * 8.0;
  }

  // Camera observations of points (sparse coupling)
  H.block<6, 3>(0, 0) = Matrix<double, 6, 3>::Random();
  H.block<6, 3>(0, 1) = Matrix<double, 6, 3>::Random();
  H.block<6, 3>(1, 2) = Matrix<double, 6, 3>::Random();
  H.block<6, 3>(1, 4) = Matrix<double, 6, 3>::Random();
  H.block<6, 3>(2, 3) = Matrix<double, 6, 3>::Random();
  H.block<6, 3>(2, 6) = Matrix<double, 6, 3>::Random();
  H.block<6, 3>(3, 5) = Matrix<double, 6, 3>::Random();
  H.block<6, 3>(3, 7) = Matrix<double, 6, 3>::Random();

  // Shared distance prior between non-consecutive points: point 1 and point 6!
  // e.g. distance constraint r = ||p1 - p6|| - d_target
  // Adds off-diagonal Hessian block between p1 and p6: J_p1^T J_p6
  H.block<3, 3>(1, 6) = Matrix<double, 3, 3>::Random() * 0.25;

  // Ground truth solution and RHS
  const VecX x_expected = VecX::Random(48);
  const VecX b = H * x_expected;

  SECTION("BlockGuidedOrdering<1> with AMD on points submatrix") {
    // In this case:
    // - Most points are independent, but points 1 and 6 are coupled by the distance prior.
    // - BlockGuidedOrdering<1> eliminates all points first (so the camera Schur complement is
    // formed).
    // - The points submatrix is automatically factored via AMD ordering on the sparse points graph!
    BlockSparseLDLT<decltype(H), BlockGuidedOrdering<1>> ldlt;
    ldlt.analyzePattern(H);
    REQUIRE(ldlt.isAnalyzed());
    REQUIRE_FALSE(ldlt.isBlockDiagonalEliminated());
    REQUIRE(ldlt.isPartitioned());
    REQUIRE(ldlt.numIndependentBlocks() == 6);
    REQUIRE(ldlt.numCoupledDimensions() == 6);
    REQUIRE(ldlt.numReduced() == 24);
    REQUIRE(ldlt.numEliminated() == 24);

    ldlt.compute(H);
    REQUIRE(ldlt.isComputed());
    REQUIRE(ldlt.info() == Eigen::Success);

    const VecX x_sol = ldlt.solve(b);
    REQUIRE((x_sol - x_expected).norm() < 1e-10);

    // Matrix solve to recover partial inverse for cameras
    const Matrix<double, Dynamic, Dynamic> I_cams =
        Matrix<double, Dynamic, Dynamic>::Identity(48, 48).leftCols(24);
    const Matrix<double, Dynamic, Dynamic> cov_cams = ldlt.solve(I_cams);
    REQUIRE((H.toDense() * cov_cams - I_cams).norm() < 1e-10);
  }

  SECTION("BlockAMDOrdering on entire system") {
    BlockSparseLDLT<decltype(H), BlockAMDOrdering> ldlt(H, 24);
    REQUIRE(ldlt.isComputed());
    REQUIRE_FALSE(ldlt.isBlockDiagonalEliminated());

    const VecX x_sol = ldlt.solve(b);
    REQUIRE((x_sol - x_expected).norm() < 1e-10);
  }
}

TEST_CASE("block_sparse_ldlt repeated compute with zero reallocation", "[solver][ldlt]") {
  BlockSparseMat<6, 3> H(2, 3);
  for (Index i = 0; i < 2; ++i) {
    Matrix<double, 6, 6> B = Matrix<double, 6, 6>::Random();
    H.diagonal<6>(i) = B * B.transpose() + Matrix<double, 6, 6>::Identity() * 10.0;
  }
  for (Index j = 0; j < 3; ++j) {
    Matrix<double, 3, 3> C = Matrix<double, 3, 3>::Random();
    H.diagonal<3>(j) = C * C.transpose() + Matrix<double, 3, 3>::Identity() * 5.0;
  }
  H.block<6, 3>(0, 0) = Matrix<double, 6, 3>::Random();
  H.block<6, 3>(1, 2) = Matrix<double, 6, 3>::Random();

  BlockSparseLDLT<decltype(H), BlockGuidedOrdering<1>> ldlt;
  ldlt.analyzePattern(H);
  ldlt.compute(H);
  REQUIRE(ldlt.isComputed());

  // Numerical values change in the problem
  H.diagonal<6>(0) += Matrix<double, 6, 6>::Identity() * 2.0;
  H.diagonal<3>(1) += Matrix<double, 3, 3>::Identity() * 1.5;
  H.block<6, 3>(0, 0) *= 0.5;

  const VecX x_new = VecX::Random(21);
  const VecX b_new = H * x_new;

  // Re-compute without calling analyzePattern again
  ldlt.compute(H);
  REQUIRE(ldlt.isComputed());

  const VecX x_sol = ldlt.solve(b_new);
  REQUIRE((x_sol - x_new).norm() < 1e-10);
}

TEST_CASE("block_sparse_ldlt singular detection and non-finite rhs", "[solver][ldlt]") {
  BlockSparseMat<6, 3> H(2, 3);
  for (Index i = 0; i < 2; ++i) {
    H.diagonal<6>(i) = Matrix<double, 6, 6>::Identity() * 10.0;
  }
  for (Index j = 0; j < 3; ++j) {
    H.diagonal<3>(j) = Matrix<double, 3, 3>::Identity() * 5.0;
  }
  H.block<6, 3>(0, 0).setConstant(1.0);

  // Singular eliminated block
  BlockSparseMat<6, 3> H_singular = H;
  H_singular.diagonal<3>(1).setZero();

  BlockSparseLDLT<decltype(H_singular), BlockGuidedOrdering<1>> ldlt_singular(H_singular);
  REQUIRE(ldlt_singular.info() != Eigen::Success);
  REQUIRE_FALSE(ldlt_singular.isComputed());

  // Non-finite RHS
  BlockSparseLDLT<decltype(H), BlockGuidedOrdering<1>> ldlt(H);
  REQUIRE(ldlt.isComputed());
  VecX b_nan = VecX::Zero(21);
  b_nan(3) = std::numeric_limits<double>::quiet_NaN();
  VecX x_nan = ldlt.solve(b_nan);
  REQUIRE(std::isnan(x_nan(0)));
  REQUIRE_FALSE(ldlt.Solve(b_nan).has_value());
}

TEST_CASE("block_sparse_ldlt parallel Schur and matrix solves are deterministic",
          "[solver][ldlt][openmp]") {
  constexpr Index kCameras = 8;
  constexpr Index kPoints = 512;
  BlockSparseMat<6, 3> H(kCameras, kPoints);
  for (Index camera = 0; camera < kCameras; ++camera) {
    H.diagonal<6>(camera) = Matrix<double, 6, 6>::Identity() * 100.0;
  }
  for (Index point = 0; point < kPoints; ++point) {
    H.diagonal<3>(point) = Matrix<double, 3, 3>::Identity() * 10.0;
    const Index camera = point % kCameras;
    Matrix<double, 6, 3> jacobian;
    for (Index row = 0; row < jacobian.rows(); ++row) {
      for (Index col = 0; col < jacobian.cols(); ++col) {
        jacobian(row, col) = 0.001 * static_cast<double>(1 + row + col + point % 7);
      }
    }
    H.block<6, 3>(camera, point) = jacobian;
    H.block<6, 3>((camera + 1) % kCameras, point) = jacobian * 0.5;
  }

  BlockSparseLDLT<decltype(H), BlockGuidedOrdering<1>> serial;
  serial.set_num_threads(1);
  serial.analyzePattern(H);
  serial.compute(H);
  REQUIRE(serial.isComputed());

  BlockSparseLDLT<decltype(H), BlockGuidedOrdering<1>> parallel;
  parallel.set_num_threads(4);
  parallel.analyzePattern(H);
  parallel.compute(H);
  REQUIRE(parallel.isComputed());
  REQUIRE((serial.schurMatrix().array() == parallel.schurMatrix().array()).all());

  const VecX rhs = VecX::LinSpaced(H.rows(), -1.0, 1.0);
  REQUIRE((serial.solve(rhs).array() == parallel.solve(rhs).array()).all());

  const Matrix<double, Dynamic, Dynamic> multiple_rhs =
      Matrix<double, Dynamic, Dynamic>::Random(H.rows(), 20);
  const auto serial_solution = serial.solve(multiple_rhs);
  const auto parallel_solution = parallel.solve(multiple_rhs);
  REQUIRE((serial_solution.array() == parallel_solution.array()).all());
}

TEST_CASE("block_sparse_ldlt parallel solve with tridiagonal eliminated blocks", "[solver][ldlt]") {
  constexpr Index kEliminatedBlocks = 20;
  BlockSparseMat<6, 3> H(2, kEliminatedBlocks);
  for (Index camera = 0; camera < 2; ++camera)
    H.diagonal<6>(camera) = Matrix<double, 6, 6>::Identity() * 100.0;
  for (Index point = 0; point < kEliminatedBlocks; ++point) {
    H.diagonal<3>(point) = Matrix<double, 3, 3>::Identity() * 10.0;
    H.block<6, 3>(point % 2, point) = Matrix<double, 6, 3>::Constant(0.01);
    if (point + 1 < kEliminatedBlocks)
      H.block<3, 3>(point, point + 1) = Matrix<double, 3, 3>::Identity() * 0.1;
  }

  BlockSparseLDLT<decltype(H), BlockGuidedOrdering<1>> serial(H);
  BlockSparseLDLT<decltype(H), BlockGuidedOrdering<1>> parallel;
  parallel.set_num_threads(4);
  parallel.analyzePattern(H);
  parallel.compute(H);
  REQUIRE(serial.isEliminatedTridiagonal());
  REQUIRE(parallel.isEliminatedTridiagonal());
  REQUIRE(parallel.info() == Eigen::Success);

  const Matrix<double, Dynamic, Dynamic> identity =
      Matrix<double, Dynamic, Dynamic>::Identity(H.rows(), H.cols());
  const Matrix<double, Dynamic, Dynamic> serial_inverse = serial.solve(identity);
  const Matrix<double, Dynamic, Dynamic> parallel_inverse = parallel.solve(identity);
  REQUIRE((parallel_inverse.array() == serial_inverse.array()).all());
}
