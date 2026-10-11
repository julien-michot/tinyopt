// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#if CATCH2_VERSION == 2
#include <catch2/catch.hpp>
#else
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#endif

#include <tinyopt/tinyopt.h>

using namespace tinyopt;
using Catch::Approx;

TEST_CASE("tinyopt_block_sparse_matrix_traits", "[block_sparse_matrix]") {
  STATIC_REQUIRE(traits::is_block_sparse_matrix_v<BlockSparseMatrix<double, 6, 3>>);
  STATIC_REQUIRE(traits::is_block_sparse_matrix_v<BlockSparseMatrix<float, 3>>);
  STATIC_REQUIRE(traits::is_block_sparse_matrix_v<BlockSparseMatix<double, 3, 2, 1>>);
  STATIC_REQUIRE(!traits::is_block_sparse_matrix_v<MatX>);
  STATIC_REQUIRE(!traits::is_block_sparse_matrix_v<SparseMat>);
}

TEST_CASE("tinyopt_block_sparse_matrix_single_size", "[block_sparse_matrix]") {
  // Matrix with single block size 3, sized for 4 blocks -> 12x12
  BlockSparseMatrix<double, 3> mat(4);

  REQUIRE(mat.num_block_types() == 1);
  REQUIRE(mat.num_blocks<3>() == 4);
  REQUIRE(mat.block_rows() == 4);
  REQUIRE(mat.block_cols() == 4);
  REQUIRE(mat.rows() == 12);
  REQUIRE(mat.cols() == 12);
  REQUIRE(mat.size() == 144);
  REQUIRE(mat.empty());

  // Set diagonal blocks
  mat.block<3, 3>(0, 0).setIdentity();
  mat.block<3, 3>(1, 1).setConstant(2.0);
  mat.block(2, 2).setConstant(3.0);  // Without template args for N=1

  // Set off-diagonal block (upper triangular)
  Mat3 B01;
  B01 << 1, 2, 3, 4, 5, 6, 7, 8, 9;
  mat.block<3, 3>(0, 1) = B01;

  REQUIRE(mat.num_allocated_blocks() == 4);
  REQUIRE(!mat.empty());

  // Access via lower-triangular coordinates (1, 0)
  auto b10 = mat.block<3, 3>(1, 0);
  REQUIRE((b10 - B01.transpose()).norm() == Approx(0.0).margin(1e-12));

  // Write to lower-triangular block (1, 0)
  Mat3 B_new;
  B_new << 9, 8, 7, 6, 5, 4, 3, 2, 1;
  mat.block<3, 3>(1, 0) = B_new;

  // Upper triangular block (0, 1) should now equal B_new.transpose()
  REQUIRE((mat.block<3, 3>(0, 1) - B_new.transpose()).norm() == Approx(0.0).margin(1e-12));

  // Modify in-place using Eigen block methods
  mat.block<3, 3>(1, 0).block<1, 1>(0, 0)(0, 0) = 42.0;
  REQUIRE(mat.block<3, 3>(0, 1)(0, 0) == Approx(42.0));

  // Diagonal access
  REQUIRE(mat.diagonal<3>(0)(0, 0) == Approx(1.0));
  REQUIRE(mat.diagonal<3>(1)(0, 0) == Approx(2.0));
  REQUIRE(mat.diagonal<3>(2)(0, 0) == Approx(3.0));

  // Whole matrix diagonal
  VecX diag = mat.diagonal();
  REQUIRE(diag.size() == 12);
  REQUIRE(diag(0) == Approx(1.0));
  REQUIRE(diag(1) == Approx(1.0));
  REQUIRE(diag(2) == Approx(1.0));
  REQUIRE(diag(3) == Approx(2.0));
  REQUIRE(diag(6) == Approx(3.0));
  REQUIRE(diag(9) == Approx(0.0));  // block 3 not allocated yet

  // Dense and Sparse conversions
  MatX dense = mat.toDense();
  REQUIRE(dense.rows() == 12);
  REQUIRE(dense.cols() == 12);
  // Check symmetry
  REQUIRE((dense - dense.transpose()).norm() == Approx(0.0).margin(1e-12));

  SparseMat sparse = mat.toSparse();
  MatX dense_sp(sparse);
  REQUIRE((dense - dense_sp).norm() == Approx(0.0).margin(1e-12));

  // Matrix-vector multiplication
  VecX x = VecX::Random(12);
  VecX y_block = mat * x;
  VecX y_dense = dense * x;
  REQUIRE((y_block - y_dense).norm() == Approx(0.0).margin(1e-12));
}

TEST_CASE("tinyopt_block_sparse_matrix_two_sizes", "[block_sparse_matrix]") {
  // Sized: 2 blocks of 6 (camera poses), 3 blocks of 3 (3D landmarks)
  // Total size: 2 * 6 + 3 * 3 = 12 + 9 = 21
  BlockSparseMatrix<double, 6, 3> mat(2, 3);

  REQUIRE(mat.num_block_types() == 2);
  REQUIRE(mat.num_blocks<6>() == 2);
  REQUIRE(mat.num_blocks<3>() == 3);
  REQUIRE(mat.rows() == 21);
  REQUIRE(mat.cols() == 21);

  // Set diagonal camera block
  Mat6 H_cam0 = Mat6::Identity() * 5.0;
  mat.block<6, 6>(0, 0) = H_cam0;

  // Set diagonal landmark block
  Mat3 H_lm1 = Mat3::Identity() * 2.0;
  mat.block<3, 3>(1, 1) = H_lm1;

  // Set camera-landmark off-diagonal block (6x3)
  Matrix<double, 6, 3> J_0_1;
  J_0_1.setRandom();
  mat.block<6, 3>(0, 1) = J_0_1;

  REQUIRE(mat.has_block<6, 3>(0, 1));
  REQUIRE(mat.has_block<3, 6>(1, 0));
  REQUIRE(!mat.has_block<6, 3>(1, 2));

  // Read via transposed view block<3, 6>(1, 0)
  auto b_trans = mat.block<3, 6>(1, 0);
  REQUIRE((b_trans - J_0_1.transpose()).norm() == Approx(0.0).margin(1e-12));

  // Assign through transposed view
  Matrix<double, 3, 6> J_new = Matrix<double, 3, 6>::Random();
  mat.block<3, 6>(1, 0) = J_new;

  // Underlying block<6, 3>(0, 1) should now be J_new.transpose()
  REQUIRE((mat.block<6, 3>(0, 1) - J_new.transpose()).norm() == Approx(0.0).margin(1e-12));

  // Test in-place arithmetic on transposed view
  mat.block<3, 6>(1, 0) += J_new;
  REQUIRE((mat.block<6, 3>(0, 1) - (2.0 * J_new.transpose())).norm() == Approx(0.0).margin(1e-12));

  // Test Eigen block methods on the view
  mat.block<3, 6>(1, 0).row(0).setZero();
  REQUIRE(mat.block<6, 3>(0, 1).col(0).norm() == Approx(0.0).margin(1e-12));

  mat.block<3, 6>(1, 0).block<2, 2>(1, 1).setConstant(7.5);
  REQUIRE(mat.block<6, 3>(0, 1).block<2, 2>(1, 1)(0, 0) == Approx(7.5));

  // Const access
  const auto& cmat = mat;
  auto cb63 = cmat.block<6, 3>(0, 1);
  auto cb36 = cmat.block<3, 6>(1, 0);
  REQUIRE((cb36 - cb63.transpose()).norm() == Approx(0.0).margin(1e-12));

  // Non-existent block access in const context returns zero block
  auto cb_empty = cmat.block<6, 3>(1, 0);
  REQUIRE(cb_empty.norm() == Approx(0.0).margin(1e-12));

  // Fast block vectors and indices access
  const auto& b63_vec = mat.blocks<6, 3>();
  const auto& b63_indices = mat.block_indices<6, 3>();
  REQUIRE(b63_vec.size() == 1);
  REQUIRE(b63_indices.size() == 1);
  REQUIRE(b63_indices[0].row == 0);
  REQUIRE(b63_indices[0].col == 1);
  REQUIRE(mat.block_index<6, 3>(0, 1) == 0);
  REQUIRE(mat.block_index<6, 3>(1, 2) == -1);

  // Conversions
  MatX dense = mat.toDense();
  REQUIRE((dense - dense.transpose()).norm() == Approx(0.0).margin(1e-12));

  SparseMat sparse = mat.toSparse();
  MatX dense_sp(sparse);
  REQUIRE((dense - dense_sp).norm() == Approx(0.0).margin(1e-12));

  // Check dense offsets: Camera 0 starts at 0, Landmark 1 starts at 2*6 + 1*3 = 15
  REQUIRE((dense.block<6, 3>(0, 15) - mat.block<6, 3>(0, 1)).norm() == Approx(0.0).margin(1e-12));
  REQUIRE((dense.block<3, 6>(15, 0) - mat.block<3, 6>(1, 0)).norm() == Approx(0.0).margin(1e-12));

  // Matrix-vector multiplication
  VecX x = VecX::Random(21);
  VecX y_block = mat * x;
  VecX y_dense = dense * x;
  REQUIRE((y_block - y_dense).norm() == Approx(0.0).margin(1e-12));

  // Norm and squaredNorm
  REQUIRE(mat.squaredNorm() == Approx(dense.squaredNorm()).margin(1e-10));
  REQUIRE(mat.norm() == Approx(dense.norm()).margin(1e-10));

  // Corner methods
  mat.topLeftCorner<6, 6>().setIdentity();
  REQUIRE(mat.block<6, 6>(0, 0)(0, 0) == Approx(1.0));
  REQUIRE(mat.block<6, 6>(0, 0)(0, 1) == Approx(0.0));

  // SetZero and SetIdentity
  mat.setIdentity();
  MatX id_dense = mat.toDense();
  REQUIRE((id_dense - MatX::Identity(21, 21)).norm() == Approx(0.0).margin(1e-12));

  mat.setZero();
  REQUIRE(mat.norm() == Approx(0.0).margin(1e-12));
  REQUIRE(mat.toDense().norm() == Approx(0.0).margin(1e-12));
}

TEST_CASE("tinyopt_block_sparse_matrix_three_sizes_float", "[block_sparse_matrix]") {
  // 3 block types: 6, 3, 2 with float scalar
  BlockSparseMatrix<float, 6, 3, 2> mat(2, 2, 2);
  // Total size: 2*6 + 2*3 + 2*2 = 12 + 6 + 4 = 22
  REQUIRE(mat.rows() == 22);
  REQUIRE(mat.cols() == 22);

  Matrix<float, 6, 2> B62;
  B62.setRandom();
  mat.block<6, 2>(0, 1) = B62;

  Matrix<float, 3, 2> B32;
  B32.setRandom();
  mat.block<3, 2>(1, 0) = B32;

  Matrix<float, 6, 3> B63;
  B63.setRandom();
  mat.block<6, 3>(1, 1) = B63;

  mat.block<6, 6>(0, 0).setIdentity();
  mat.block<3, 3>(0, 0).setIdentity();
  mat.block<2, 2>(1, 1).setIdentity();

  // Test lower-triangular views for all cross pairs:
  // (2, 6) views (6, 2)
  REQUIRE((mat.block<2, 6>(1, 0) - B62.transpose()).norm() == Approx(0.0f).margin(1e-6f));
  // (2, 3) views (3, 2)
  REQUIRE((mat.block<2, 3>(0, 1) - B32.transpose()).norm() == Approx(0.0f).margin(1e-6f));
  // (3, 6) views (6, 3)
  REQUIRE((mat.block<3, 6>(1, 1) - B63.transpose()).norm() == Approx(0.0f).margin(1e-6f));

  // Dense and Sparse equivalence and symmetry
  MatXf dense = mat.toDense();
  REQUIRE((dense - dense.transpose()).norm() == Approx(0.0f).margin(1e-6f));

  SparseMatf sparse = mat.toSparse();
  MatXf dense_sp(sparse);
  REQUIRE((dense - dense_sp).norm() == Approx(0.0f).margin(1e-6f));

  // Matrix-vector multiplication
  VecXf x = VecXf::Random(22);
  VecXf y_block = mat * x;
  VecXf y_dense = dense * x;
  REQUIRE((y_block - y_dense).norm() == Approx(0.0f).margin(1e-5f));
}

TEST_CASE("tinyopt_block_sparse_matrix_dynamic_growth", "[block_sparse_matrix]") {
  // Default constructed matrix without upfront sizing
  BlockSparseMatrix<double, 4, 2> mat;
  REQUIRE(mat.rows() == 0);
  REQUIRE(mat.cols() == 0);

  Matrix<double, 4, 2> B;
  B.setRandom();
  // Accessing index (1, 3) should grow block counts to at least 2 of size 4 and 4 of size 2
  mat.block<4, 2>(1, 3) = B;

  REQUIRE(mat.num_blocks<4>() == 2);
  REQUIRE(mat.num_blocks<2>() == 4);
  // Total rows = 2 * 4 + 4 * 2 = 8 + 8 = 16
  REQUIRE(mat.rows() == 16);
  REQUIRE(mat.cols() == 16);

  MatX dense = mat.toDense();
  REQUIRE((dense - dense.transpose()).norm() == Approx(0.0).margin(1e-12));

  // Check offset: type 0 starts at 0, block 1 starts at 1*4 = 4.
  // type 1 starts at 2*4 = 8, block 3 starts at 8 + 3*2 = 14.
  REQUIRE((dense.block<4, 2>(4, 14) - B).norm() == Approx(0.0).margin(1e-12));
  REQUIRE((dense.block<2, 4>(14, 4) - B.transpose()).norm() == Approx(0.0).margin(1e-12));
}

TEST_CASE("tinyopt_block_sparse_matrix_alias_spelling", "[block_sparse_matrix]") {
  // Test BlockSparseMatix alias
  BlockSparseMatix<double, 3, 2> mat(1, 1);
  mat.block<3, 2>(0, 0).setConstant(1.23);
  REQUIRE(mat.block<3, 2>(0, 0)(0, 0) == Approx(1.23));
  REQUIRE(mat.block<2, 3>(0, 0)(0, 0) == Approx(1.23));

  // Test BlockSparseMat (double) and BlockSparseMatf (float)
  BlockSparseMat<6, 3> mat_d(1, 2);
  STATIC_REQUIRE(std::is_same_v<typename decltype(mat_d)::ScalarType, double>);
  mat_d.block<6, 3>(0, 1).setConstant(4.56);
  REQUIRE(mat_d.block<6, 3>(0, 1)(0, 0) == Approx(4.56));
  REQUIRE(mat_d.block<3, 6>(1, 0)(0, 0) == Approx(4.56));

  BlockSparseMatf<6, 3> mat_f(1, 2);
  STATIC_REQUIRE(std::is_same_v<typename decltype(mat_f)::ScalarType, float>);
  mat_f.block<6, 3>(0, 1).setConstant(7.89f);
  REQUIRE(mat_f.block<6, 3>(0, 1)(0, 0) == Approx(7.89f));
  REQUIRE(mat_f.block<3, 6>(1, 0)(0, 0) == Approx(7.89f));
}
