// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cassert>
#include <optional>

#include <Eigen/SparseCholesky>
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_LU)
#include <Eigen/SparseLU>
#endif
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_QR)
#include <Eigen/SparseQR>
#endif
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_INCOMPLETE_CHOLESKY)
#include <Eigen/IterativeLinearSolvers>
#endif

#include <tinyopt/math/solvers/base.h>
#include <tinyopt/math/solvers/dense.h>
#if defined(TINYOPT_ENABLE_SUITESPARSE)
#include <tinyopt/math/suitesparse.h>
#endif

namespace tinyopt {

template <typename BlockMatrix>
struct BlockSparseLinearSolvers;

template <typename MatrixType>
struct SparseLinearSolvers {
  using Scalar = typename MatrixType::Scalar;
  using StorageIndex = typename MatrixType::StorageIndex;
  using VectorType = Vector<Scalar, Dynamic>;
  using DenseMatrix = Matrix<Scalar, Dynamic, Dynamic>;
  using DenseVector = Vector<Scalar, Dynamic>;
  using ReducedSolver = DenseLinearSolvers<DenseMatrix>;

  Index dims = 0;
  LinearSolverOptions options{};

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
  bool schur_enabled_ = false;
  Index num_reduced_ = 0;

  explicit SparseLinearSolvers(LinearSolverMethod m = LinearSolverMethod::LDLT, Index d = 0,
                               const LinearSolverOptions &opt = {})
      : SparseLinearSolvers(OptionsWithMethod(opt, m), d) {}

  explicit SparseLinearSolvers(const LinearSolverOptions &opt, Index d = 0)
      : dims(d), options(opt), reduced_schur_solver(ReducedOptions(opt)) {
    if (dims > 0) {
      resize(dims);
    }
  }

  void reset() {
    is_analyzed_ = false;
    schur_enabled_ = false;
    num_reduced_ = 0;
    reduced_schur_solver.reset();
  }

  void resize(Index new_dims, Index num_reduced = 0) {
    if (dims != new_dims || num_reduced_ != num_reduced) {
      dims = new_dims;
      num_reduced_ = num_reduced;
      schur_enabled_ = num_reduced > 0;
      is_analyzed_ = false;
      if (schur_enabled_) reduced_schur_solver.resize(num_reduced);
    }
  }

  [[nodiscard]] bool IsAnalyzed() const { return is_analyzed_; }

  template <typename InputMatrixType>
  bool Analyze(const InputMatrixType &H, Index num_reduced = 0) {
    if (H.rows() != H.cols() || num_reduced < 0 || num_reduced >= H.rows() ||
        (num_reduced > 0 && !options.allow_schur)) {
      is_analyzed_ = false;
      schur_enabled_ = false;
      num_reduced_ = -1;
      return false;
    }
    resize(H.rows(), num_reduced);
    if (num_reduced > 0) {
      const Index num_eliminated = H.rows() - num_reduced;
      return AnalyzePattern(H.bottomRightCorner(num_eliminated, num_eliminated));
    }
    return AnalyzePattern(H);
  }

 private:
  template <typename InputMatrixType>
  bool AnalyzePattern(const InputMatrixType &H) {
    is_analyzed_ = false;
    switch (ActiveMethod()) {
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
    return is_analyzed_;
  }

 public:
  template <typename InputMatrixType>
  bool Compute(const InputMatrixType &A) {
    assert(A.rows() == A.cols());
    switch (ActiveMethod()) {
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

 private:
  template <typename DerivedB>
  Eigen::Matrix<Scalar, DerivedB::RowsAtCompileTime, DerivedB::ColsAtCompileTime> SolverLinear(
      const MatrixBase<DerivedB> &b) const {
    using Result = Eigen::Matrix<Scalar, DerivedB::RowsAtCompileTime, DerivedB::ColsAtCompileTime>;
    switch (ActiveMethod()) {
      case LinearSolverMethod::LDLT: {
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_LDLT)
        return ldlt_.solve(b);
#endif
        break;
      }
      case LinearSolverMethod::LLT: {
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_LLT)
        return llt_.solve(b);
#endif
        break;
      }
      case LinearSolverMethod::LU: {
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_LU)
        return lu_.solve(b);
#endif
        break;
      }
      case LinearSolverMethod::QR: {
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_QR)
        return qr_.solve(b);
#endif
        break;
      }
      case LinearSolverMethod::IncompleteCholesky: {
#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_INCOMPLETE_CHOLESKY)
        return incomplete_cholesky_.solve(b);
#endif
        break;
      }
      default:
        break;
    }
    return Result::Zero(b.rows(), b.cols());
  }

 public:
  Eigen::ComputationInfo Info() const {
    switch (ActiveMethod()) {
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

 private:
  template <typename InputMatrixType, typename DerivedB>
  std::optional<Vector<Scalar, DerivedB::RowsAtCompileTime>> SolverLinear(
      const InputMatrixType &A, const MatrixBase<DerivedB> &b) {
    assert(A.rows() == A.cols());
    assert(A.rows() == b.rows());
    using Result = Vector<Scalar, DerivedB::RowsAtCompileTime>;

    if (options.method == LinearSolverMethod::SuiteSparse && !schur_enabled_) {
#if defined(TINYOPT_ENABLE_SUITESPARSE)
      return suitesparse_detail::Solve<Scalar, DerivedB::RowsAtCompileTime>(A, b);
#else
      return std::nullopt;
#endif
    }

    if (!Compute(A)) return std::nullopt;
    Result solution = SolverLinear(b);
    if (Info() != Eigen::Success || !solution.allFinite()) return std::nullopt;
    return solution;
  }

 public:
  [[nodiscard]] LinearSolverMethod GetMethod() const { return options.method; }
  [[nodiscard]] Index Dims() const { return dims; }

  template <typename InputMatrixType, typename DerivedB>
  std::optional<DenseVector> Solve(const InputMatrixType &H, const MatrixBase<DerivedB> &rhs) {
    if (num_reduced_ < 0 || num_reduced_ >= H.rows() ||
        (num_reduced_ > 0 && !options.allow_schur)) {
      return std::nullopt;
    }
    if (num_reduced_ == 0) {
      const auto solution = SolverLinear(H, rhs);
      if (!solution) return std::nullopt;
      return DenseVector(*solution);
    }

    return SolveSchur(H, rhs);
  }

  template <typename InputMatrixType>
  std::optional<MatrixType> Inverse(const InputMatrixType &H) {
    if (H.rows() != H.cols() || H.rows() <= 0) return std::nullopt;

    // When Schur reduction is active the cached symbolic analysis describes the eliminated
    // bottom-right block (size rows - num_reduced), not the full Hessian. Factorize a temporary
    // solver on the full matrix so the Schur state is left untouched for subsequent solves.
    if (schur_enabled_ || num_reduced_ > 0) {
      Eigen::SimplicialLDLT<MatrixType, Eigen::Upper> full_solver;
      full_solver.compute(H);
      if (full_solver.info() != Eigen::Success) return std::nullopt;

      MatrixType identity(H.rows(), H.cols());
      identity.setIdentity();

      MatrixType inv = full_solver.solve(identity);
      if (full_solver.info() != Eigen::Success) return std::nullopt;
      return inv;
    }

    if (!Compute(H)) return std::nullopt;

    MatrixType identity(H.rows(), H.cols());
    identity.setIdentity();

    MatrixType inv = ldlt_.solve(identity);
    if (Info() != Eigen::Success) return std::nullopt;
    return inv;
  }

 private:
  template <typename InputMatrixType, typename DerivedB>
  std::optional<DenseVector> SolveSchur(const InputMatrixType &H, const MatrixBase<DerivedB> &rhs) {
    const Index dimension = H.rows();
    if (!options.allow_schur || H.cols() != dimension || rhs.size() != dimension ||
        num_reduced_ <= 0 || num_reduced_ >= dimension || !rhs.allFinite()) {
      return std::nullopt;
    }
    if (!is_analyzed_ && !Analyze(H, num_reduced_)) return std::nullopt;
    const Index num_eliminated = dimension - num_reduced_;
    SparseMatrix<Scalar> eliminated = H.bottomRightCorner(num_eliminated, num_eliminated);
    if (!Compute(eliminated)) return std::nullopt;

    const auto coupling = H.topRightCorner(num_reduced_, num_eliminated);
    const DenseVector solved_eliminated_rhs = SolverLinear(DenseVector(rhs.tail(num_eliminated)));
    if (Info() != Eigen::Success || !solved_eliminated_rhs.allFinite()) {
      return std::nullopt;
    }

    DenseMatrix reduced = H.topLeftCorner(num_reduced_, num_reduced_);
    CompleteSymmetricMatrix(reduced);
    // Stream columns through the sparse factorization instead of storing A^-1 C^T.
    for (Index col = 0; col < num_reduced_; ++col) {
      const DenseVector coupling_column = coupling.transpose().col(col);
      const DenseVector solved_coupling_column = SolverLinear(coupling_column);
      if (Info() != Eigen::Success || !solved_coupling_column.allFinite()) {
        return std::nullopt;
      }
      reduced.col(col).noalias() -= coupling * solved_coupling_column;
    }
    if (!reduced.allFinite()) return std::nullopt;

    DenseVector reduced_rhs = rhs.head(num_reduced_) - coupling * solved_eliminated_rhs;
    const auto reduced_solution = reduced_schur_solver.Solve(reduced, reduced_rhs);
    if (!reduced_solution) return std::nullopt;

    DenseVector solution(dimension);
    solution.head(num_reduced_) = *reduced_solution;
    const DenseVector reduced_coupling_rhs = coupling.transpose() * solution.head(num_reduced_);
    const DenseVector solved_reduced_coupling = SolverLinear(reduced_coupling_rhs);
    if (Info() != Eigen::Success || !solved_reduced_coupling.allFinite()) {
      return std::nullopt;
    }
    solution.tail(num_eliminated) = solved_eliminated_rhs - solved_reduced_coupling;
    if (!solution.allFinite()) return std::nullopt;
    return solution;
  }

 private:
  static LinearSolverOptions OptionsWithMethod(LinearSolverOptions solver_options,
                                               LinearSolverMethod solver_method) {
    solver_options.method = solver_method;
    return solver_options;
  }

  static LinearSolverOptions ReducedOptions(LinearSolverOptions solver_options) {
    if (solver_options.method == LinearSolverMethod::SuiteSparse ||
        solver_options.method == LinearSolverMethod::IncompleteCholesky) {
      solver_options.method = LinearSolverMethod::LDLT;
    }
    return solver_options;
  }

  LinearSolverMethod ActiveMethod() const {
    if (schur_enabled_ && options.method == LinearSolverMethod::SuiteSparse) {
      return LinearSolverMethod::LDLT;
    }
    return options.method;
  }

  template <typename>
  friend struct BlockSparseLinearSolvers;

  ReducedSolver reduced_schur_solver;
};

}  // namespace tinyopt
