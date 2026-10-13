// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cassert>
#include <cmath>
#include <optional>

#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_LDLT) || defined(TINYOPT_ENABLE_LINEAR_SOLVER_LLT)
#include <Eigen/Cholesky>
#endif
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_LU)
#include <Eigen/LU>
#endif
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_QR)
#include <Eigen/QR>
#endif
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_SVD)
#include <Eigen/SVD>
#endif

#include <tinyopt/math/small_linear_solver.h>
#include <tinyopt/math/solvers/base.h>

namespace tinyopt {

template <typename BlockMatrix>
struct BlockSparseLinearSolvers;

template <typename MatrixType>
struct DenseLinearSolvers {
  using Scalar = typename MatrixType::Scalar;
  static constexpr int RowsAtCompileTime = MatrixType::RowsAtCompileTime;
  static constexpr int ColsAtCompileTime = MatrixType::ColsAtCompileTime;
  using VectorType = Vector<Scalar, RowsAtCompileTime>;
  using DenseMatrix = Matrix<Scalar, Dynamic, Dynamic>;
  using DenseVector = Vector<Scalar, Dynamic>;

  Index dims = 0;
  LinearSolverOptions options{};

#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_LDLT)
  Eigen::LDLT<MatrixType, Eigen::Upper> ldlt_;
#endif
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_LLT)
  Eigen::LLT<MatrixType, Eigen::Upper> llt_;
#endif
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_LU)
  Eigen::PartialPivLU<MatrixType> lu_;
#endif
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_QR)
  Eigen::ColPivHouseholderQR<MatrixType> qr_;
#endif
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_SVD)
  Eigen::JacobiSVD<MatrixType, Eigen::ComputeFullU | Eigen::ComputeFullV> svd_;
#endif

  explicit DenseLinearSolvers(LinearSolverMethod m = LinearSolverMethod::LDLT,
                              Index d = (RowsAtCompileTime != Dynamic ? RowsAtCompileTime : 0),
                              const LinearSolverOptions &opt = {})
      : dims(d), options(opt) {
    options.method = m;
    if (dims > 0) {
      resize(dims);
    }
  }

  explicit DenseLinearSolvers(const LinearSolverOptions &opt,
                              Index d = (RowsAtCompileTime != Dynamic ? RowsAtCompileTime : 0))
      : dims(d), options(opt) {
    if (dims > 0) {
      resize(dims);
    }
  }

  void resize(Index new_dims, Index num_reduced = 0) {
    dims = new_dims;
    num_reduced_ = num_reduced;
    schur_enabled_ = num_reduced > 0;
    if constexpr (RowsAtCompileTime != Dynamic) {
      return;
    }
    const Index factor_dims = schur_enabled_ ? dims - num_reduced_ : dims;
    if (factor_dims <= 0) return;

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

    switch (options.method) {
      case LinearSolverMethod::LDLT:
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_LDLT)
        ldlt_ = Eigen::LDLT<MatrixType, Eigen::Upper>(factor_dims);
#endif
        break;
      case LinearSolverMethod::LLT:
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_LLT)
        llt_ = Eigen::LLT<MatrixType, Eigen::Upper>(factor_dims);
#endif
        break;
      case LinearSolverMethod::LU:
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_LU)
        lu_ = Eigen::PartialPivLU<MatrixType>(factor_dims);
#endif
        break;
      case LinearSolverMethod::QR:
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_QR)
        qr_ = Eigen::ColPivHouseholderQR<MatrixType>(factor_dims, factor_dims);
#endif
        break;
      case LinearSolverMethod::SVD:
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_SVD)
        svd_ = Eigen::JacobiSVD<MatrixType>(factor_dims, factor_dims,
                                            Eigen::ComputeFullU | Eigen::ComputeFullV);
#endif
        break;
      default:
        break;
    }

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
  }

  void reset() {
    num_reduced_ = 0;
    schur_enabled_ = false;
  }

  template <typename DerivedA>
  bool Analyze(const MatrixBase<DerivedA> &H, Index num_reduced = 0) {
    if (num_reduced < 0 || num_reduced >= H.rows() || H.rows() != H.cols() ||
        (num_reduced > 0 && (!options.allow_schur || RowsAtCompileTime != Dynamic))) {
      num_reduced_ = -1;
      schur_enabled_ = false;
      return false;
    }
    resize(H.rows(), num_reduced);
    return true;
  }

 private:
  template <typename DerivedA>
  bool Compute(const MatrixBase<DerivedA> &A) {
    static_assert(DerivedA::RowsAtCompileTime == Dynamic ||
                      DerivedA::ColsAtCompileTime == Dynamic ||
                      DerivedA::RowsAtCompileTime == DerivedA::ColsAtCompileTime,
                  "DenseLinearSolvers requires a square matrix");
    assert(A.rows() == A.cols());
    switch (options.method) {
      case LinearSolverMethod::LDLT: {
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_LDLT)
        ldlt_.compute(A);
        if (ldlt_.info() != Eigen::Success) return false;
        const auto epsilon = Eigen::NumTraits<Scalar>::epsilon();
        const auto d_min = ldlt_.vectorD().minCoeff();
        const auto d_max = ldlt_.vectorD().maxCoeff();
        return d_max > Scalar(0) && d_min > epsilon * d_max;
#else
        return false;
#endif
      }
      case LinearSolverMethod::LLT: {
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_LLT)
        llt_.compute(A);
        return llt_.info() == Eigen::Success;
#else
        return false;
#endif
      }
      case LinearSolverMethod::LU: {
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_LU)
        lu_.compute(A);
        const auto epsilon = Eigen::NumTraits<Scalar>::epsilon();
        return lu_.rcond() > epsilon;
#else
        return false;
#endif
      }
      case LinearSolverMethod::QR: {
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_QR)
        qr_.compute(A);
        return qr_.info() == Eigen::Success && qr_.rank() == A.cols();
#else
        return false;
#endif
      }
      case LinearSolverMethod::SVD: {
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_SVD)
        if (!std::isfinite(options.svd_relative_threshold) || options.svd_relative_threshold < 0.0)
          return false;
        svd_.compute(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
        if (options.svd_relative_threshold > 0.0) {
          svd_.setThreshold(static_cast<Scalar>(options.svd_relative_threshold));
        }
        return true;
#else
        return false;
#endif
      }
      default:
        return false;
    }
  }

  template <typename DerivedB>
  std::optional<Vector<Scalar, DerivedB::RowsAtCompileTime>> SolverLinear(
      const MatrixBase<DerivedB> &b) const {
    using Result = Vector<Scalar, DerivedB::RowsAtCompileTime>;
    Result solution;
    switch (options.method) {
      case LinearSolverMethod::LDLT: {
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_LDLT)
        if (ldlt_.info() != Eigen::Success) return std::nullopt;
        const auto epsilon = Eigen::NumTraits<Scalar>::epsilon();
        const auto d_min = ldlt_.vectorD().minCoeff();
        const auto d_max = ldlt_.vectorD().maxCoeff();
        if (!(d_max > Scalar(0) && d_min > epsilon * d_max)) return std::nullopt;
        solution = ldlt_.solve(b);
        if (ldlt_.info() != Eigen::Success || !solution.allFinite()) return std::nullopt;
        return solution;
#else
        return std::nullopt;
#endif
      }
      case LinearSolverMethod::LLT: {
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_LLT)
        if (llt_.info() != Eigen::Success) return std::nullopt;
        solution = llt_.solve(b);
        if (llt_.info() != Eigen::Success || !solution.allFinite()) return std::nullopt;
        return solution;
#else
        return std::nullopt;
#endif
      }
      case LinearSolverMethod::LU: {
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_LU)
        solution = lu_.solve(b);
        if (!solution.allFinite()) return std::nullopt;
        return solution;
#else
        return std::nullopt;
#endif
      }
      case LinearSolverMethod::QR: {
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_QR)
        if (qr_.info() != Eigen::Success) return std::nullopt;
        solution = qr_.solve(b);
        if (!solution.allFinite()) return std::nullopt;
        return solution;
#else
        return std::nullopt;
#endif
      }
      case LinearSolverMethod::SVD: {
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_SVD)
        solution = svd_.solve(b);
        if (!solution.allFinite()) return std::nullopt;
        return solution;
#else
        return std::nullopt;
#endif
      }
      default:
        return std::nullopt;
    }
  }

  template <typename DerivedA, typename DerivedB>
  std::optional<Vector<Scalar, DerivedB::RowsAtCompileTime>> SolverLinear(
      const MatrixBase<DerivedA> &A, const MatrixBase<DerivedB> &b) {
    static_assert(DerivedA::RowsAtCompileTime == Dynamic ||
                      DerivedA::ColsAtCompileTime == Dynamic ||
                      DerivedA::RowsAtCompileTime == DerivedA::ColsAtCompileTime,
                  "DenseLinearSolvers requires a square matrix");
    assert(A.rows() == A.cols());
    assert(A.rows() == b.rows());

    if constexpr (DerivedA::RowsAtCompileTime == DerivedA::ColsAtCompileTime &&
                  DerivedA::RowsAtCompileTime >= 1 && DerivedA::RowsAtCompileTime <= 4) {
      return SolveSmallDenseSystem(A, b);
    } else {
      if (!Compute(A)) return std::nullopt;
      return SolverLinear(b);
    }
  }

 public:
  [[nodiscard]] LinearSolverMethod GetMethod() const { return options.method; }
  [[nodiscard]] Index Dims() const { return dims; }

  template <typename DerivedA, typename DerivedB>
  std::optional<DenseVector> Solve(const MatrixBase<DerivedA> &H, const MatrixBase<DerivedB> &rhs) {
    if (num_reduced_ < 0 || num_reduced_ >= H.rows() ||
        (num_reduced_ > 0 && (!options.allow_schur || RowsAtCompileTime != Dynamic))) {
      return std::nullopt;
    }
    if (num_reduced_ == 0) {
      const auto solution = SolverLinear(H, rhs);
      if (!solution) return std::nullopt;
      return DenseVector(*solution);
    }

    return SolveSchur(H, rhs);
  }

 private:
  template <typename DerivedA, typename DerivedB>
  std::optional<DenseVector> SolveSchur(const MatrixBase<DerivedA> &H,
                                        const MatrixBase<DerivedB> &rhs) {
    const Index dimension = H.rows();
    if (!options.allow_schur || H.cols() != dimension || rhs.size() != dimension ||
        num_reduced_ <= 0 || num_reduced_ >= dimension || !rhs.allFinite()) {
      return std::nullopt;
    }
    const Index num_eliminated = dimension - num_reduced_;
    DenseMatrix eliminated = H.bottomRightCorner(num_eliminated, num_eliminated);
    if (!Compute(eliminated)) return std::nullopt;

    const DenseMatrix coupling = H.topRightCorner(num_reduced_, num_eliminated);
    DenseMatrix solved_coupling(num_eliminated, num_reduced_);
    for (Index col = 0; col < num_reduced_; ++col) {
      const auto solved_column = SolverLinear(coupling.transpose().col(col));
      if (!solved_column) return std::nullopt;
      solved_coupling.col(col) = *solved_column;
    }

    const auto solved_eliminated_rhs = SolverLinear(rhs.tail(num_eliminated));
    if (!solved_eliminated_rhs) return std::nullopt;

    DenseMatrix reduced = H.topLeftCorner(num_reduced_, num_reduced_);
    CompleteSymmetricMatrix(reduced);
    reduced.noalias() -= coupling * solved_coupling;
    if (!reduced.allFinite()) return std::nullopt;

    DenseVector reduced_rhs = rhs.head(num_reduced_) - coupling * *solved_eliminated_rhs;
    if (!Compute(reduced)) return std::nullopt;
    const auto reduced_solution = SolverLinear(reduced_rhs);
    if (!reduced_solution) return std::nullopt;

    DenseVector solution(dimension);
    solution.head(num_reduced_) = *reduced_solution;
    solution.tail(num_eliminated) =
        *solved_eliminated_rhs - solved_coupling * solution.head(num_reduced_);
    if (!solution.allFinite()) return std::nullopt;
    return solution;
  }

  bool schur_enabled_ = false;
  Index num_reduced_ = 0;

  template <typename>
  friend struct BlockSparseLinearSolvers;
};

}  // namespace tinyopt
