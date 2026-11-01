// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <limits>

#include <Eigen/Dense>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <tinyopt/math/block_tridiag_ldlt.h>
#include <tinyopt/math/block_tridiag_matrix.h>
#include <tinyopt/types.h>

using namespace tinyopt;
using Eigen::MatrixXd;

TEST_CASE("BlockTridiagMatrix construction and dense conversion", "[tridiagonal][matrix]") {
  constexpr int B = 3;
  constexpr Index N = 5;
  BlockTridiagMat<B> T(N);

  REQUIRE(T.num_blocks() == N);
  REQUIRE(T.block_size() == B);
  REQUIRE(T.rows() == N * B);
  REQUIRE(T.cols() == N * B);

  // Fill diagonal and superdiagonal blocks
  for (Index i = 0; i < N; ++i) {
    Matrix<double, B, B> D = Matrix<double, B, B>::Random();
    T.diagonal(i) = D * D.transpose() + Matrix<double, B, B>::Identity() * 5.0;
  }
  for (Index i = 0; i < N - 1; ++i) {
    T.off_diagonal(i) = Matrix<double, B, B>::Random();
  }

  // Verify dense matrix symmetry and values
  const MatrixXd dense = T.toDense();
  REQUIRE(dense.rows() == N * B);
  REQUIRE(dense.cols() == N * B);
  REQUIRE((dense - dense.transpose()).norm() < 1e-7);

  // Check block access
  for (Index i = 0; i < N; ++i) {
    Matrix<double, B, B> d_dense = dense.block<B, B>(i * B, i * B);
    REQUIRE((d_dense - T.diagonal(i)).norm() < 1e-7);
  }
  for (Index i = 0; i < N - 1; ++i) {
    Matrix<double, B, B> u_dense = dense.block<B, B>(i * B, (i + 1) * B);
    REQUIRE((u_dense - T.off_diagonal(i)).norm() < 1e-7);
    Matrix<double, B, B> l_dense = dense.block<B, B>((i + 1) * B, i * B);
    REQUIRE((l_dense - T.off_diagonal(i).transpose()).norm() < 1e-7);
  }

  // Check matrix-vector product
  const VecX x = VecX::Random(N * B);
  const VecX y_tridiag = T * x;
  const VecX y_dense = dense * x;
  REQUIRE((y_tridiag - y_dense).norm() < 1e-7);

  // Check scaleDiagonal (Levenberg-Marquardt damping)
  T.scaleDiagonal(2.0);
  const MatrixXd damped_dense = T.toDense();
  for (Index i = 0; i < N; ++i) {
    for (int r = 0; r < B; ++r) {
      REQUIRE(Catch::Approx(damped_dense(i * B + r, i * B + r)) ==
              2.0 * dense(i * B + r, i * B + r));
    }
  }
}

namespace {

/// Compares the block tridiagonal LDLT against a dense LDLT for one block size.
template <int B>
void CheckTridiagLDLT(int actual_B) {
  const Index N = 6;
  const Index dims = N * actual_B;

  BlockTridiagMatrix<double, B> T(N, actual_B);

  for (Index i = 0; i < N; ++i) {
    MatrixXd D = MatrixXd::Random(actual_B, actual_B);
    T.diagonal(i) = D * D.transpose() + MatrixXd::Identity(actual_B, actual_B) * 10.0;
  }
  for (Index i = 0; i < N - 1; ++i) {
    T.off_diagonal(i) = MatrixXd::Random(actual_B, actual_B) * 0.5;
  }

  const MatrixXd dense = T.toDense();
  const Eigen::LDLT<MatrixXd> dense_ldlt(dense);
  REQUIRE(dense_ldlt.info() == Eigen::Success);

  BlockTridiagLDLT<double, B> tridiag_ldlt;
  tridiag_ldlt.analyzePattern(T);
  REQUIRE(tridiag_ldlt.isAnalyzed());

  tridiag_ldlt.compute(T);
  REQUIRE(tridiag_ldlt.isComputed());
  REQUIRE(tridiag_ldlt.info() == Eigen::Success);

  // Vector RHS solve
  {
    const VecX b = VecX::Random(dims);
    const VecX x_dense = dense_ldlt.solve(b);
    const VecX x_tridiag = tridiag_ldlt.solve(b);

    REQUIRE((x_tridiag - x_dense).norm() < 1e-6);

    const auto opt_sol = tridiag_ldlt.Solve(b);
    REQUIRE(opt_sol.has_value());
    REQUIRE((*opt_sol - x_dense).norm() < 1e-6);
  }

  // Matrix RHS solve (inverse recovery)
  {
    const MatrixXd I = MatrixXd::Identity(dims, dims);
    const MatrixXd inv_dense = dense_ldlt.solve(I);
    const MatrixXd inv_tridiag = tridiag_ldlt.solve(I);

    REQUIRE((inv_tridiag - inv_dense).norm() < 1e-6);
    REQUIRE((dense * inv_tridiag - I).norm() < 1e-6);
  }

  // Zero-copy computeFromPointers
  {
    std::vector<const typename BlockTridiagLDLT<double, B>::Block*> diag_ptrs(
        static_cast<size_t>(N));
    std::vector<const typename BlockTridiagLDLT<double, B>::Block*> off_diag_ptrs(
        static_cast<size_t>(N - 1));

    for (Index i = 0; i < N; ++i) {
      diag_ptrs[static_cast<size_t>(i)] = &T.diagonal(i);
    }
    for (Index i = 0; i < N - 1; ++i) {
      off_diag_ptrs[static_cast<size_t>(i)] = &T.off_diagonal(i);
    }

    BlockTridiagLDLT<double, B> ptr_ldlt;
    ptr_ldlt.analyzePattern(N, actual_B);
    ptr_ldlt.computeFromPointers(diag_ptrs, off_diag_ptrs);
    REQUIRE(ptr_ldlt.isComputed());
    REQUIRE(ptr_ldlt.info() == Eigen::Success);

    const VecX b = VecX::Random(dims);
    const VecX x_ptr = ptr_ldlt.solve(b);
    const VecX x_dense = dense_ldlt.solve(b);
    REQUIRE((x_ptr - x_dense).norm() < 1e-6);
  }
}

}  // namespace

using Case2 = std::integral_constant<int, 2>;
using Case4 = std::integral_constant<int, 4>;
using CaseDyn = std::integral_constant<int, Dynamic>;

TEMPLATE_TEST_CASE("BlockTridiagLDLT vector and matrix solve against dense LDLT",
                   "[tridiagonal][ldlt]", Case2, Case4, CaseDyn) {
  constexpr int B = TestType::value;
  if constexpr (B == Dynamic) {
    // Exercise the odd and large fixed block sizes without extra template instantiations.
    for (const int block_size : {1, 3, 5, 6}) {
      CheckTridiagLDLT<B>(block_size);
    }
  } else {
    CheckTridiagLDLT<B>(B);
  }
}
