// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tinyopt/types.h>
#include <Eigen/src/SVD/JacobiSVD.h>

namespace tinyopt {

/**
 * @brief Computes the inverse of a dense, symmetric, semi-definite matrix.
 *
 * This function calculates the inverse of a symmetric matrix, commonly used for covariance or
 * information matrices. It leverages the LDLT decomposition for efficiency and numerical stability.
 *
 * @param[in] m The dense symmetric input matrix. It can be filled either fully or only in the upper
 * triangular part.
 *
 * @return The inverse of the input matrix or nullopt, with the same dimensions and scalar type as
 * the input.
 *
 * @note The input matrix is assumed to be symmetric. If only the upper triangular part is filled,
 * the function implicitly uses the symmetry to construct the full matrix for the LDLT
 * decomposition.
 *
 * @tparam Scalar The scalar type of the matrix elements.
 *
 * @code
 * Eigen::MatrixXd covarianceMatrix(3, 3);
 * covarianceMatrix << 1.0, 0.5, 0.2,
 * 0.5, 2.0, 0.8,
 * 0.2, 0.8, 3.0;
 *
 * auto inverseCovariance = InvCov(covarianceMatrix);
 * if (inverseCovariance)
 *   std::cout << "Inverse Covariance Matrix:\n" << inverseCovariance.value() << std::endl;
 * @endcode
 */
template <typename Derived>
std::optional<
    Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>>
DenseInvCov(const MatrixBase<Derived> &m) {
  using MatType =
      Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>;
  if constexpr (Derived::ColsAtCompileTime == 1) {
    return MatType(m.cwiseInverse().asDiagonal());
  } else if (m.cols() == 1) {
    return MatType(m.inverse());  // un-protected here...
  } else if (m.cols() > 0) {
    const auto &chol = m.template selfadjointView<Upper>().ldlt();
    if (chol.info() == Eigen::Success && chol.isPositive())
      return chol.solve(MatType::Identity(m.rows(), m.cols()));
  }
  return std::nullopt;
}

/**
 * @brief Computes the inverse of a dense, symmetric, semi-definite matrix.
 *
 * This function calculates the inverse of a symmetric matrix, commonly used for covariance or
 * information matrices. It leverages the LDLT decomposition for efficiency and numerical stability.
 *
 * @param[in] m The dense symmetric input matrix. It can be filled either fully or only in the upper
 * triangular part.
 *
 * @return The inverse of the input matrix or nullopt, with the same dimensions and scalar type as
 * the input.
 *
 * @note The input matrix is assumed to be symmetric. If only the upper triangular part is filled,
 * the function implicitly uses the symmetry to construct the full matrix for the LDLT
 * decomposition.
 *
 * @tparam Scalar The scalar type of the matrix elements.
 *
 * @code
 * Eigen::MatrixXd covarianceMatrix(3, 3);
 * covarianceMatrix << 1.0, 0.5, 0.2,
 * 0.5, 2.0, 0.8,
 * 0.2, 0.8, 3.0;
 *
 * auto inverseCovariance = InvCov(covarianceMatrix);
 * if (inverseCovariance)
 *   std::cout << "Inverse Covariance Matrix:\n" << inverseCovariance.value() << std::endl;
 * @endcode
 */
template <typename Derived>
auto InvCov(const MatrixBase<Derived> &m) {
  return DenseInvCov(m);
}

/**
 * @brief Computes the inverse of a sparse, symmetric, semi-definite matrix.
 *
 * This function calculates the inverse of a symmetric matrix, commonly used for covariance or
 * information matrices. It leverages the LDLT decomposition for efficiency and numerical stability.
 *
 * @param[in] m The symmetric input matrix. It can be filled either fully or only in the upper
 * triangular part.
 *
 * @param[in] retry_with_shift_offset On numerical failure, the function will try again with the
 * specified shift offset, only if the value is not 0, e.g. 1e-4.
 *
 * @return The inverse of the input matrix or nullopt, with the same dimensions and scalar type as
 * the input.
 *
 * @note The input matrix is assumed to be symmetric. If only the upper triangular part is filled,
 * the function implicitly uses the symmetry to construct the full matrix for the LDLT
 * decomposition.
 *
 * @tparam Scalar The scalar type of the matrix elements.
 */
template <typename T>
std::optional<SparseMatrix<typename T::Scalar>> SparseInvCov(
    const T &m, typename T::Scalar retry_with_shift_offset = typename T::Scalar(0.0)) {
  using Scalar = typename T::Scalar;
  if (m.size() == 0) return std::nullopt;
  Eigen::SimplicialLDLT<SparseMatrix<Scalar>, Eigen::Upper> solver;
  solver.compute(m);
  if (solver.info() != Eigen::Success) {  // Decomposition failed
    if (retry_with_shift_offset > 0) {
      if (solver.info() == Eigen::NumericalIssue) {
        solver.setShift(retry_with_shift_offset);
        solver.compute(m);
      }
    }
    if (solver.info() != Eigen::Success) return std::nullopt;
  }
  SparseMatrix<Scalar> I(m.rows(), m.cols());
  I.setIdentity();
  auto X = solver.solve(I);
  if (solver.info() != Eigen::Success) {  // Solving failed
    return std::nullopt;
  }
  return X;
}

/**
 * @brief Computes the inverse of a sparse, symmetric, semi-definite matrix.
 *
 * This function calculates the inverse of a symmetric matrix, commonly used for covariance or
 * information matrices. It leverages the LDLT decomposition for efficiency and numerical stability.
 *
 * @param[in] m The symmetric input matrix. It can be filled either fully or only in the upper
 * triangular part.
 *
 * @param[in] retry_with_shift_offset On numerical failure, the function will try again with the
 * specified shift offset, only if the value is not 0, e.g. 1e-4.
 *
 * @return The inverse of the input matrix or nullopt, with the same dimensions and scalar type as
 * the input.
 *
 * @note The input matrix is assumed to be symmetric. If only the upper triangular part is filled,
 * the function implicitly uses the symmetry to construct the full matrix for the LDLT
 * decomposition.
 *
 * @tparam Scalar The scalar type of the matrix elements.
 */
template <typename Scalar>
std::optional<SparseMatrix<Scalar>> InvCov(const SparseMatrix<Scalar> &m,
                                           Scalar retry_with_shift_offset = Scalar(0.0)) {
  return SparseInvCov(m, retry_with_shift_offset);
}

/**
 * @brief Computes the inverse of a dense or sparse, symmetric, semi-definite matrix.
 *
 * This function calculates the inverse of a symmetric matrix, commonly used for covariance or
 * information matrices. It leverages the LDLT decomposition for efficiency and numerical stability.
 *
 * @param[in] m The dense or sparse input matrix. It can be filled either fully or only in the upper
 * triangular part.
 *
 * @return The inverse of the input matrix or nullopt, with the same dimensions and scalar type as
 * the input.
 *
 * @note The input matrix is assumed to be symmetric. If only the upper triangular part is filled,
 * the function implicitly uses the symmetry to construct the full matrix for the LDLT
 * decomposition.
 */
template <typename XprType, int BlockRows, int BlockCols, bool InnerPanel>
auto InvCov(const Block<XprType, BlockRows, BlockCols, InnerPanel> &m) {
  using Scalar = typename XprType::Scalar;
  if constexpr (std::is_same_v<std::decay_t<XprType>, SparseMatrix<Scalar>>)
    return SparseInvCov(m);
  else
    return DenseInvCov(m);
}

/**
 * @brief Solves the linear system A * X = B for X, where A is a symmetric positive-definite matrix,
 * return optional otherwise.
 *
 * This function utilizes the LDLT decomposition (Cholesky decomposition) to efficiently solve the
 * linear system. It assumes that A is a symmetric positive-definite matrix, which is a requirement
 * for the LDLT decomposition.
 *
 * @tparam Derived The type of the matrix A, which must be a square, symmetric, and
 * positive-definite matrix.
 * @tparam Derived2 The type of the vector B.
 *
 * @param A The coefficient matrix A.
 * @param b The right-hand side vector B.
 *
 * @return An `std::optional` containing the solution vector X if the system is solvable (A is
 * positive-definite), or `std::nullopt` if the system is not solvable (A is not positive-definite).
 *
 * @note The function returns `chol.solve(b)`, which implies that the solution returned is actually
 * the solution of A * X = B.
 *
 * @throws (Implicitly) Throws exceptions from Eigen if the LDLT decomposition fails due to reasons
 * other than non-positive definiteness (e.g., memory allocation failure).
 *
 * @code
 * Eigen::MatrixXd A = Eigen::MatrixXd::Random(3, 3);
 * A = A * A.transpose(); // Ensure A is symmetric positive-definite (or semi-definite)
 * Eigen::VectorXd b = Eigen::VectorXd::Random(3);
 *
 * auto x_opt = Solve(A, b);
 *
 * if (x_opt) {
 *  Eigen::VectorXd x = x_opt.value();
 *  std::cout << "Solution X: " << x.transpose() << std::endl;
 *  std::cout << "A * X: " << (A * x).transpose() << std::endl;
 *  std::cout << "Expected -B: " << b.transpose() << std::endl;
 * } else {
 *  std::cout << "Matrix A is not positive-definite. System cannot be solved." << std::endl;
 * }
 * @endcode
 */
template <typename Derived, typename Derived2>
std::optional<Vector<typename Derived::Scalar, Derived::RowsAtCompileTime>> SolveLDLT(
    const MatrixBase<Derived> &A, const MatrixBase<Derived2> &b) {
  const auto &chol = A.template selfadjointView<Upper>().ldlt();
  if (chol.info() == Eigen::Success && chol.isPositive()) {
    return chol.solve(b);
  }
  return std::nullopt;
}

/**
 * @brief Solves the linear system A * X = B for X, where A is a symmetric positive-definite sparse
 * matrix, return optional otherwise.
 *
 * This function utilizes the LDLT decomposition (Cholesky decomposition) to efficiently solve the
 * linear system. It assumes that A is a symmetric positive-definite matrix, which is a requirement
 * for the LDLT decomposition.
 *
 * @tparam Derived The type of the matrix A, which must be a square, symmetric, and
 * positive-definite matrix.
 * @tparam Derived2 The type of the vector B.
 *
 * @param A The sparse coefficient matrix A.
 * @param b The right-hand side vector B.
 *
 * @return An `std::optional` containing the solution vector X if the system is solvable (A is
 * positive-definite), or `std::nullopt` if the system is not solvable (A is not positive-definite).
 *
 * @note The function returns `chol.solve(b)`, which implies that the solution returned is actually
 * the solution of A * X = B.
 *
 * @throws (Implicitly) Throws exceptions from Eigen if the LDLT decomposition fails due to reasons
 * other than non-positive definiteness (e.g., memory allocation failure).
 */
template <typename Scalar, int RowsAtCompileTime = Dynamic>
std::optional<Vector<Scalar, RowsAtCompileTime>> SolveLDLT(
    const SparseMatrix<Scalar> &A, const Vector<Scalar, RowsAtCompileTime> &b) {
  Eigen::SimplicialLDLT<SparseMatrix<Scalar>, Eigen::Upper> solver;
  solver.compute(A);
  if (solver.info() != Eigen::Success)  // Decomposition failed
    return std::nullopt;
  auto X = solver.solve(b);
  if (solver.info() != Eigen::Success)  // Solving failed
    return std::nullopt;
  return X;
}

/**
 * @brief Solves the linear system A * X = B for X using LU decomposition.
 *
 * @tparam Derived The type of the matrix A.
 * @tparam Derived2 The type of the vector B.
 *
 * @param A The coefficient matrix A.
 * @param b The right-hand side vector B.
 *
 * @return An `std::optional` containing the solution vector X if the system is solvable, or
 * `std::nullopt` otherwise.
 */
template <typename Derived, typename Derived2>
std::optional<Vector<typename Derived::Scalar, Derived::RowsAtCompileTime>> SolveLU(
    const MatrixBase<Derived> &A, const MatrixBase<Derived2> &b) {
  auto lu = A.partialPivLu();
  if (lu.determinant() != 0) {
    return lu.solve(b);
  }
  return std::nullopt;
}

/**
 * @brief Solves the linear system A * X = B for X using QR decomposition.
 *
 * @tparam Derived The type of the matrix A.
 * @tparam Derived2 The type of the vector B.
 *
 * @param A The coefficient matrix A.
 * @param b The right-hand side vector B.
 *
 * @return An `std::optional` containing the solution vector X.
 */
template <typename Derived, typename Derived2>
std::optional<Vector<typename Derived::Scalar, Derived::RowsAtCompileTime>> SolveQR(
    const MatrixBase<Derived> &A, const MatrixBase<Derived2> &b) {
  return A.colPivHouseholderQr().solve(b);
}

/**
 * @brief Solves the linear system A * X = B for X using SVD decomposition.
 *
 * @tparam Derived The type of the matrix A.
 * @tparam Derived2 The type of the vector B.
 *
 * @param A The coefficient matrix A.
 * @param b The right-hand side vector B.
 *
 * @return An `std::optional` containing the solution vector X.
 */
template <typename Derived, typename Derived2>
std::optional<Vector<typename Derived::Scalar, Derived::RowsAtCompileTime>> SolveSVD(
    const MatrixBase<Derived> &A, const MatrixBase<Derived2> &b) {
  return A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
}


/**
 * @brief Solves the sparse linear system A * X = B for X using LU decomposition.
 *
 * @tparam Scalar The scalar type.
 * @tparam RowsAtCompileTime The compile-time number of rows of vector B.
 *
 * @param A The sparse coefficient matrix A.
 * @param b The right-hand side vector B.
 *
 * @return An `std::optional` containing the solution vector X if the system is solvable, or
 * `std::nullopt` otherwise.
 */
template <typename Scalar, int RowsAtCompileTime = Dynamic>
std::optional<Vector<Scalar, RowsAtCompileTime>> SolveLU(
    const SparseMatrix<Scalar> &A, const Vector<Scalar, RowsAtCompileTime> &b) {
  Eigen::SparseLU<SparseMatrix<Scalar>> solver;
  solver.compute(A);
  if (solver.info() != Eigen::Success) return std::nullopt;
  auto X = solver.solve(b);
  if (solver.info() != Eigen::Success) return std::nullopt;
  return X;
}

/**
 * @brief Solves the sparse linear system A * X = B for X using QR decomposition.
 *
 * @tparam Scalar The scalar type.
 * @tparam RowsAtCompileTime The compile-time number of rows of vector B.
 *
 * @param A The sparse coefficient matrix A.
 * @param b The right-hand side vector B.
 *
 * @return An `std::optional` containing the solution vector X if the system is solvable, or
 * `std::nullopt` otherwise.
 */
template <typename Scalar, int RowsAtCompileTime = Dynamic>
std::optional<Vector<Scalar, RowsAtCompileTime>> SolveQR(
    const SparseMatrix<Scalar> &A, const Vector<Scalar, RowsAtCompileTime> &b) {
  Eigen::SparseQR<SparseMatrix<Scalar>, Eigen::COLAMDOrdering<int>> solver;
  solver.compute(A);
  if (solver.info() != Eigen::Success) return std::nullopt;
  auto X = solver.solve(b);
  if (solver.info() != Eigen::Success) return std::nullopt;
  return X;
}


/// Integer square root function for positive integers
/// Will return `N` for negative or 0 values
constexpr inline int SQRT(int N) {
  if (N <= 1) return N;
  int left = 1, right = N / 2;
  int result = 0;
  while (left <= right) {
    int mid = left + (right - left) / 2;
    if (mid <= N / mid) {
      left = mid + 1;
      result = mid;
    } else {
      right = mid - 1;
    }
  }
  return result;
};

template <typename Scalar = double>
inline constexpr Scalar FloatEpsilon() {
  /*static*/ const Scalar eps = static_cast<Scalar>(std::is_same_v<Scalar, float> ? 1e-4f : 1e-7f);
  return eps;
}

template <typename Scalar = double>
inline constexpr Scalar FloatEpsilon2() {
  /*static*/ const Scalar eps = static_cast<Scalar>(std::is_same_v<Scalar, float> ? 1e-8f : 1e-14f);
  return eps;
}

/// A constexpr version of the ternary operator: (condition) ? ValueOnTrue : ValueOnFalse
#ifndef _MSC_VER  // due to error C3493...
#define If(condition, ValueOnTrue, ValueOnFalse) \
  [&]() {                                        \
    if constexpr (condition)                     \
      return ValueOnTrue;                        \
    else                                         \
      return ValueOnFalse;                       \
  }()
#endif

// Function to compute the maximum absolute difference between two sparse matrices
template <typename T>
T MaxAbsDiff(const Eigen::SparseMatrix<T> &mat1, const Eigen::SparseMatrix<T> &mat2) {
  // Check if matrices have the same dimensions
  if (mat1.rows() != mat2.rows() || mat1.cols() != mat2.cols()) {
    throw std::invalid_argument("Matrices must have the same dimensions.");
  }

  T maxDiff = 0;
  // Iterate through the non-zero elements of both matrices.  This is more efficient
  // for sparse matrices.
  for (int k = 0; k < mat1.outerSize(); ++k) {
    for (typename Eigen::SparseMatrix<T>::InnerIterator it1(mat1, k); it1; ++it1) {
      T val1 = it1.value();
      T val2 = 0;  // Default value if the element is not present in mat2

      // Try to find the corresponding element in mat2.  This is the
      // crucial part for efficiency with sparse matrices.  We do NOT
      // iterate through all of mat2.
      for (typename Eigen::SparseMatrix<T>::InnerIterator it2(mat2, k); it2; ++it2) {
        if (it2.row() == it1.row()) {
          val2 = it2.value();
          break;  // Important: Exit the inner loop once found
        }
      }
      T diff = std::abs(val1 - val2);
      maxDiff = std::max(maxDiff, diff);
    }
  }

  // Check for elements present in mat2 but not in mat1.  This is necessary
  // because the previous loop only iterated through non-zeros in mat1.
  for (int k = 0; k < mat2.outerSize(); ++k) {
    for (typename Eigen::SparseMatrix<T>::InnerIterator it2(mat2, k); it2; ++it2) {
      T val2 = it2.value();
      T val1 = 0;
      bool found = false;
      for (typename Eigen::SparseMatrix<T>::InnerIterator it1(mat1, k); it1; ++it1) {
        if (it1.row() == it2.row()) {
          val1 = it1.value();
          found = true;
          break;
        }
      }
      if (!found) {  // if the element was not found in mat1
        T diff = std::abs(val1 - val2);
        maxDiff = std::max(maxDiff, diff);
      }
    }
  }
  return maxDiff;
}
}  // namespace tinyopt