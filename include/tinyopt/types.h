// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <Eigen/src/Core/util/Constants.h>
#include <Eigen/src/SparseCore/SparseSelfAdjointView.h>
#include <Eigen/SparseCholesky>

namespace tinyopt {

static constexpr int Dynamic = Eigen::Dynamic;
static constexpr int Lower = Eigen::Lower;
static constexpr int Upper = Eigen::Upper;
static constexpr int Infinity = Eigen::Infinity;

using Index = Eigen::Index;

template <typename Scalar, int Rows = Dynamic, int Cols = Dynamic, int Options = 0,
          int MaxRows = Rows, int MaxCols = Cols>
using Matrix =
    std::conditional_t<Rows != 1 || (Rows == 1 && Cols == 1),
                       Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>,
                       Eigen::Matrix<Scalar, Rows, Cols, Eigen::RowMajor, MaxRows, MaxCols>>;

template <typename Scalar, int Rows = Dynamic>
using Vector = Matrix<Scalar, Rows, 1>;

template <typename T, int Cols>
using RowVector = Matrix<T, 1, Cols, Eigen::RowMajor>;

template <typename Scalar = double, int Options = 0, typename StorageIndex = int>
using SparseMatrix = Eigen::SparseMatrix<Scalar, Options, StorageIndex>;

using SparseMatrixf = SparseMatrix<float>;

template <typename XprType, int BlockRows, int BlockCols, bool InnerPanel>
using Block = Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>;
template <typename T>
using MatrixBase = Eigen::MatrixBase<T>;
template <typename T>
using DenseBase = Eigen::DenseBase<T>;
template <typename T>
using ArrayBase = Eigen::ArrayBase<T>;

// Let's define some common matrix and vectors
using MatX = Matrix<double, Dynamic, Dynamic>;
using MatXf = Matrix<float, Dynamic, Dynamic>;

using Mat1 = Matrix<double, 1, 1>;
using Mat2 = Matrix<double, 2, 2>;
using Mat3 = Matrix<double, 3, 3>;
using Mat4 = Matrix<double, 4, 4>;
using Mat5 = Matrix<double, 5, 5>;
using Mat6 = Matrix<double, 6, 6>;

using Mat1f = Matrix<float, 1, 1>;
using Mat2f = Matrix<float, 2, 2>;
using Mat3f = Matrix<float, 3, 3>;
using Mat4f = Matrix<float, 4, 4>;
using Mat5f = Matrix<float, 5, 5>;
using Mat6f = Matrix<float, 6, 6>;

using VecX = Vector<double, Dynamic>;
using Vec1 = Vector<double, 1>;
using Vec2 = Vector<double, 2>;
using Vec3 = Vector<double, 3>;
using Vec4 = Vector<double, 4>;
using Vec5 = Vector<double, 5>;
using Vec6 = Vector<double, 6>;
using Vec12 = Vector<double, 12>;
using Vec21 = Vector<double, 21>;

using VecXf = Vector<float, Dynamic>;
using Vec1f = Vector<float, 1>;
using Vec2f = Vector<float, 2>;
using Vec3f = Vector<float, 3>;
using Vec4f = Vector<float, 4>;
using Vec5f = Vector<float, 5>;
using Vec6f = Vector<float, 6>;
using Vec12f = Vector<float, 12>;
using Vec21f = Vector<float, 21>;

using Mat2X = Matrix<double, 2, Dynamic>;
using Mat3X = Matrix<double, 3, Dynamic>;

using Mat2Xf = Matrix<float, 2, Dynamic>;
using Mat3Xf = Matrix<float, 3, Dynamic>;

using Mat23 = Matrix<double, 2, 3>;
using Mat32 = Matrix<double, 3, 2>;

using Mat23f = Matrix<float, 2, 3>;
using Mat32f = Matrix<float, 3, 2>;

}  // namespace tinyopt