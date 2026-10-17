// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <optional>

#include <Eigen/SparseCholesky>

#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_LDLT) || defined(TINYOPT_ENABLE_LINEAR_SOLVER_LLT)
#include <Eigen/Cholesky>
#endif
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_LU)
#include <Eigen/LU>
#include <Eigen/SparseLU>
#endif
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_QR)
#include <Eigen/QR>
#include <Eigen/SparseQR>
#endif
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_SVD)
#include <Eigen/SVD>
#endif
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_INCOMPLETE_CHOLESKY)
#include <Eigen/IterativeLinearSolvers>
#endif

#include <tinyopt/math/small_linear_solver.h>
#include <tinyopt/types.h>

#if defined(TINYOPT_ENABLE_SUITESPARSE)
#include <tinyopt/math/suitesparse.h>
#endif

namespace tinyopt {

enum class LinearSolverMethod : uint8_t {
  LDLT,
  LLT,
  LU,
  QR,
  SVD,
  SuiteSparse,  // Cholmod from SuiteSparse
  IncompleteCholesky,
};

constexpr bool RequiresFullMatrix(LinearSolverMethod method) {
  return method != LinearSolverMethod::LDLT && method != LinearSolverMethod::LLT &&
         method != LinearSolverMethod::SuiteSparse &&
         method != LinearSolverMethod::IncompleteCholesky;
}

template <typename Derived>
void CompleteSymmetricMatrix(MatrixBase<Derived> &matrix) {
  matrix.derived() = matrix.derived().template selfadjointView<Eigen::Upper>();
}

template <typename Scalar, int Options, typename StorageIndex>
void CompleteSymmetricMatrix(SparseMatrix<Scalar, Options, StorageIndex> &matrix) {
  SparseMatrix<Scalar, Options, StorageIndex> full =
      matrix.template selfadjointView<Eigen::Upper>();
  matrix.swap(full);
}

struct LinearSolverOptions {
  LinearSolverMethod method = LinearSolverMethod::LDLT;
  /// Relative singular-value cutoff for SVD; 0 uses Eigen's default threshold.
  double svd_relative_threshold = 0.0;
  /// Useful for iterative solvers that update H in place
  bool reuse_symbolic = true;
  /// Initial shift parameter for IncompleteCholesky factorization
  double initial_shift = 1e-3;
};

struct LinearSolverBase {
  LinearSolverMethod method = LinearSolverMethod::LDLT;
  Index dims = 0;
  LinearSolverOptions options{};

  constexpr LinearSolverBase() = default;
  constexpr LinearSolverBase(LinearSolverMethod m, Index d, const LinearSolverOptions &opt)
      : method(m), dims(d), options(opt) {
    options.method = m;
  }
  constexpr LinearSolverBase(const LinearSolverOptions &opt, Index d)
      : method(opt.method), dims(d), options(opt) {}

  [[nodiscard]] LinearSolverMethod GetMethod() const { return method; }
  [[nodiscard]] Index Dims() const { return dims; }
};

template <typename MatrixType>
struct DenseLinearSolvers : public LinearSolverBase {
  using Scalar = typename MatrixType::Scalar;
  static constexpr int RowsAtCompileTime = MatrixType::RowsAtCompileTime;
  static constexpr int ColsAtCompileTime = MatrixType::ColsAtCompileTime;
  using VectorType = Vector<Scalar, RowsAtCompileTime>;

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
      : LinearSolverBase(m, d, opt) {
    if (dims > 0) {
      resize(dims);
    }
  }

  explicit DenseLinearSolvers(const LinearSolverOptions &opt,
                              Index d = (RowsAtCompileTime != Dynamic ? RowsAtCompileTime : 0))
      : LinearSolverBase(opt, d) {
    if (dims > 0) {
      resize(dims);
    }
  }

  void resize(Index new_dims) {
    dims = new_dims;
    if constexpr (RowsAtCompileTime != Dynamic) {
      return;
    }
    if (dims <= 0) return;
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

    switch (method) {
      case LinearSolverMethod::LDLT:
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_LDLT)
        ldlt_ = Eigen::LDLT<MatrixType, Eigen::Upper>(dims);
#endif
        break;
      case LinearSolverMethod::LLT:
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_LLT)
        llt_ = Eigen::LLT<MatrixType, Eigen::Upper>(dims);
#endif
        break;
      case LinearSolverMethod::LU:
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_LU)
        lu_ = Eigen::PartialPivLU<MatrixType>(dims);
#endif
        break;
      case LinearSolverMethod::QR:
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_QR)
        qr_ = Eigen::ColPivHouseholderQR<MatrixType>(dims, dims);
#endif
        break;
      case LinearSolverMethod::SVD:
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_SVD)
        svd_ = Eigen::JacobiSVD<MatrixType, Eigen::ComputeFullU | Eigen::ComputeFullV>(dims, dims);
#endif
        break;
      default:
        break;
    }
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
  }

  void reset() {}
  void analyze([[maybe_unused]] const MatrixType &H) {}

  template <typename DerivedA>
  bool compute(const MatrixBase<DerivedA> &A) {
    static_assert(DerivedA::RowsAtCompileTime == Dynamic ||
                      DerivedA::ColsAtCompileTime == Dynamic ||
                      DerivedA::RowsAtCompileTime == DerivedA::ColsAtCompileTime,
                  "DenseLinearSolvers requires a square matrix");
    assert(A.rows() == A.cols());
    switch (method) {
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
        svd_.compute(A);
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
  std::optional<Vector<Scalar, DerivedB::RowsAtCompileTime>> solve(
      const MatrixBase<DerivedB> &b) const {
    using Result = Vector<Scalar, DerivedB::RowsAtCompileTime>;
    Result solution;
    switch (method) {
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
  std::optional<Vector<Scalar, DerivedB::RowsAtCompileTime>> solve(const MatrixBase<DerivedA> &A,
                                                                   const MatrixBase<DerivedB> &b) {
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
      if (!compute(A)) return std::nullopt;
      return solve(b);
    }
  }
};

template <typename MatrixType>
struct SparseLinearSolvers : public LinearSolverBase {
  using Scalar = typename MatrixType::Scalar;
  using StorageIndex = typename MatrixType::StorageIndex;
  using VectorType = Vector<Scalar, Dynamic>;

#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_LDLT)
  Eigen::SimplicialLDLT<MatrixType, Eigen::Upper> ldlt_;
#endif
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_LLT)
  Eigen::SimplicialLLT<MatrixType, Eigen::Upper> llt_;
#endif
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_LU)
  Eigen::SparseLU<MatrixType> lu_;
#endif
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_QR)
  Eigen::SparseQR<MatrixType, Eigen::COLAMDOrdering<int>> qr_;
#endif
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_INCOMPLETE_CHOLESKY)
  Eigen::IncompleteCholesky<Scalar, Eigen::Upper, Eigen::AMDOrdering<StorageIndex>>
      incomplete_cholesky_;
#endif
  bool is_analyzed_ = false;

  explicit SparseLinearSolvers(LinearSolverMethod m = LinearSolverMethod::LDLT, Index d = 0,
                               const LinearSolverOptions &opt = {})
      : LinearSolverBase(m, d, opt) {
    if (dims > 0) {
      resize(dims);
    }
  }

  explicit SparseLinearSolvers(const LinearSolverOptions &opt, Index d = 0)
      : LinearSolverBase(opt, d) {
    if (dims > 0) {
      resize(dims);
    }
  }

  void reset() { is_analyzed_ = false; }

  void resize(Index new_dims) {
    if (dims != new_dims) {
      dims = new_dims;
      is_analyzed_ = false;
    }
  }

  [[nodiscard]] bool IsAnalyzed() const { return is_analyzed_; }

  template <typename InputMatrixType>
  void analyze(const InputMatrixType &H) {
    switch (method) {
      case LinearSolverMethod::LDLT:
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_LDLT)
        ldlt_.analyzePattern(H);
        is_analyzed_ = (ldlt_.info() == Eigen::Success);
#endif
        break;
      case LinearSolverMethod::LLT:
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_LLT)
        llt_.analyzePattern(H);
        is_analyzed_ = (llt_.info() == Eigen::Success);
#endif
        break;
      case LinearSolverMethod::LU:
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_LU)
        lu_.analyzePattern(H);
        is_analyzed_ = (lu_.info() == Eigen::Success);
#endif
        break;
      case LinearSolverMethod::QR:
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_QR)
        qr_.analyzePattern(H);
        is_analyzed_ = (qr_.info() == Eigen::Success);
#endif
        break;
      case LinearSolverMethod::IncompleteCholesky:
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_INCOMPLETE_CHOLESKY)
        incomplete_cholesky_.setInitialShift(
            static_cast<typename Eigen::NumTraits<Scalar>::Real>(options.initial_shift));
        incomplete_cholesky_.analyzePattern(H);
        is_analyzed_ = (incomplete_cholesky_.info() == Eigen::Success);
#endif
        break;
      case LinearSolverMethod::SuiteSparse:
#if defined(TINYOPT_ENABLE_SUITESPARSE)
        is_analyzed_ = true;
#endif
        break;
      default:
        break;
    }
  }

  template <typename InputMatrixType>
  bool compute(const InputMatrixType &A) {
    assert(A.rows() == A.cols());
    switch (method) {
      case LinearSolverMethod::LDLT: {
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_LDLT)
        if (is_analyzed_ && options.reuse_symbolic) {
          ldlt_.factorize(A);
        } else {
          ldlt_.compute(A);
          is_analyzed_ = (ldlt_.info() == Eigen::Success);
        }
        return ldlt_.info() == Eigen::Success;
#else
        return false;
#endif
      }
      case LinearSolverMethod::LLT: {
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_LLT)
        if (is_analyzed_ && options.reuse_symbolic) {
          llt_.factorize(A);
        } else {
          llt_.compute(A);
          is_analyzed_ = (llt_.info() == Eigen::Success);
        }
        return llt_.info() == Eigen::Success;
#else
        return false;
#endif
      }
      case LinearSolverMethod::LU: {
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_LU)
        if (is_analyzed_ && options.reuse_symbolic) {
          lu_.factorize(A);
        } else {
          lu_.compute(A);
          is_analyzed_ = (lu_.info() == Eigen::Success);
        }
        return lu_.info() == Eigen::Success;
#else
        return false;
#endif
      }
      case LinearSolverMethod::QR: {
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_QR)
        if (is_analyzed_ && options.reuse_symbolic) {
          qr_.factorize(A);
        } else {
          qr_.compute(A);
          is_analyzed_ = (qr_.info() == Eigen::Success);
        }
        return qr_.info() == Eigen::Success && qr_.rank() == A.cols();
#else
        return false;
#endif
      }
      case LinearSolverMethod::IncompleteCholesky: {
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_INCOMPLETE_CHOLESKY)
        incomplete_cholesky_.setInitialShift(
            static_cast<typename Eigen::NumTraits<Scalar>::Real>(options.initial_shift));
        if (is_analyzed_ && options.reuse_symbolic) {
          incomplete_cholesky_.factorize(A);
        } else {
          incomplete_cholesky_.compute(A);
          is_analyzed_ = (incomplete_cholesky_.info() == Eigen::Success);
        }
        return incomplete_cholesky_.info() == Eigen::Success;
#else
        return false;
#endif
      }
      case LinearSolverMethod::SuiteSparse: {
#if defined(TINYOPT_ENABLE_SUITESPARSE)
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
  Eigen::Matrix<Scalar, DerivedB::RowsAtCompileTime, DerivedB::ColsAtCompileTime> solve(
      const MatrixBase<DerivedB> &b) const {
    using Result = Eigen::Matrix<Scalar, DerivedB::RowsAtCompileTime, DerivedB::ColsAtCompileTime>;
    switch (method) {
      case LinearSolverMethod::LDLT: {
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_LDLT)
        return ldlt_.solve(b);
#else
        return Result::Zero(b.rows(), b.cols());
#endif
      }
      case LinearSolverMethod::LLT: {
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_LLT)
        return llt_.solve(b);
#else
        return Result::Zero(b.rows(), b.cols());
#endif
      }
      case LinearSolverMethod::LU: {
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_LU)
        return lu_.solve(b);
#else
        return Result::Zero(b.rows(), b.cols());
#endif
      }
      case LinearSolverMethod::QR: {
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_QR)
        return qr_.solve(b);
#else
        return Result::Zero(b.rows(), b.cols());
#endif
      }
      case LinearSolverMethod::IncompleteCholesky: {
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_INCOMPLETE_CHOLESKY)
        return incomplete_cholesky_.solve(b);
#else
        return Result::Zero(b.rows(), b.cols());
#endif
      }
      default:
        return Result::Zero(b.rows(), b.cols());
    }
  }

  Eigen::ComputationInfo info() const {
    switch (method) {
      case LinearSolverMethod::LDLT:
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_LDLT)
        return ldlt_.info();
#else
        return Eigen::InvalidInput;
#endif
      case LinearSolverMethod::LLT:
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_LLT)
        return llt_.info();
#else
        return Eigen::InvalidInput;
#endif
      case LinearSolverMethod::LU:
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_LU)
        return lu_.info();
#else
        return Eigen::InvalidInput;
#endif
      case LinearSolverMethod::QR:
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_QR)
        return qr_.info();
#else
        return Eigen::InvalidInput;
#endif
      case LinearSolverMethod::IncompleteCholesky:
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_INCOMPLETE_CHOLESKY)
        return incomplete_cholesky_.info();
#else
        return Eigen::InvalidInput;
#endif
      case LinearSolverMethod::SuiteSparse:
#if defined(TINYOPT_ENABLE_SUITESPARSE)
        return Eigen::Success;
#else
        return Eigen::InvalidInput;
#endif
      default:
        return Eigen::InvalidInput;
    }
  }

  template <typename InputMatrixType, typename DerivedB>
  std::optional<Vector<Scalar, DerivedB::RowsAtCompileTime>> solve(const InputMatrixType &A,
                                                                   const MatrixBase<DerivedB> &b) {
    assert(A.rows() == A.cols());
    assert(A.rows() == b.rows());
    using Result = Vector<Scalar, DerivedB::RowsAtCompileTime>;

    if (method == LinearSolverMethod::SuiteSparse) {
#if defined(TINYOPT_ENABLE_SUITESPARSE)
      return suitesparse_detail::Solve<Scalar, DerivedB::RowsAtCompileTime>(A, b);
#else
      return std::nullopt;
#endif
    }

    if (!compute(A)) return std::nullopt;
    Result solution = solve(b);
    if (info() != Eigen::Success || !solution.allFinite()) return std::nullopt;
    return solution;
  }
};

}  // namespace tinyopt