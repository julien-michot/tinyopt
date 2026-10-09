// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tinyopt/optimizers/optimizer.h>

namespace tinyopt {

template <typename Derived, typename Hessian_t>
class Optimizer2Base
    : public OptimizerCore<Derived, typename Hessian_t::Scalar,
                           SQRT(traits::params_trait<Hessian_t>::Dims), false, true> {
 public:
  using Scalar = typename Hessian_t::Scalar;
  static constexpr Index Dims = SQRT(traits::params_trait<Hessian_t>::Dims);
  using Grad_t = Vector<Scalar, Dims>;
  using H_t = Hessian_t;
  using Options = tinyopt::Options;
  using Base = OptimizerCore<Derived, Scalar, Dims, false, true>;

  explicit Optimizer2Base(const Options &options) : Base(NormalizeOptions(options)) {}
  virtual ~Optimizer2Base() = default;
  Optimizer2Base(const Optimizer2Base &) = default;
  Optimizer2Base &operator=(const Optimizer2Base &) = default;
  Optimizer2Base(Optimizer2Base &&) = default;
  Optimizer2Base &operator=(Optimizer2Base &&) = default;

  void InitWith(const Grad_t &gradient, const H_t &hessian) {
    grad_ = gradient;
    H_ = hessian;
  }

  void reset() {
    clear();
    use_locked_permutation_ = false;
    ResetStrategy();
  }

  template <int D = Dims, std::enable_if_t<D == Dynamic, int> = 0>
  bool resize(int dims) {
    if (dims <= 0) throw std::invalid_argument("Dimensions must be positive");
    InitLockedPermutation(dims);
    if (grad_.rows() == dims && H_.rows() == dims) return false;
    grad_.resize(dims);
    H_.resize(dims, dims);
    ResizeStrategy(dims);
    return true;
  }

  template <int D = Dims, std::enable_if_t<D != Dynamic, int> = 0>
  bool resize(int dims = Dims) {
    if (dims != Dims) throw std::invalid_argument("Static and dynamic dimensions must match");
    if constexpr (traits::is_sparse_matrix_v<H_t>) {
      grad_.resize(dims);
      H_.resize(dims, dims);
      ResizeStrategy(dims);
      return true;
    }
    return false;
  }

  void clear() {
    grad_.setZero();
    H_.setZero();
  }

  /// Hook called before optimization loop to initialize locked permutations if needed
  void InitOptimization(Index dims) { InitLockedPermutation(dims); }

  template <typename X_t>
  bool ResizeIfNeeded(const X_t &x) {
    if constexpr (Dims == Dynamic) {
      const Index dims = traits::params_trait<X_t>::dims(x);
      if (grad_.rows() != dims) return resize(dims);
    }
    return false;
  }

  template <typename X_t, typename AccFunc>
  Scalar Evaluate(const X_t &x, const AccFunc &acc, bool save) {
    std::nullptr_t null_gradient;
    H_t null_hessian;
    Cost value = acc(x, null_gradient, null_hessian);
    NormalizeCost(value);
    if (save) cost_ = value;
    return value.cost;
  }

  template <typename X_t, typename AccFunc>
  bool Accumulate(const X_t &x, const AccFunc &acc) {
    cost_ = acc(x, grad_, H_);
    NormalizeCost(cost_);
    return cost_.isValid();
  }

  template <typename X_t, typename AccFunc>
  bool Build(const X_t &x, const AccFunc &acc, bool resize_and_clear = true) {
    if (ShouldRebuildLinearSystem()) {
      if (resize_and_clear) {
        ResizeIfNeeded(x);
        clear();
      }
      if (!Accumulate(x, acc)) return false;
      this->Clamp(grad_, this->options_.opt.grad_clipping);
      if (this->options_.hessian.check_min_H_diag > 0 &&
          (H_.diagonal().cwiseAbs().array() < this->options_.hessian.check_min_H_diag).any())
        return false;
      if (!this->options_.hessian.H_is_full && RequiresFullMatrix(this->options_.linear_solver))
        CompleteSymmetricMatrix(H_);
      ApplyJacobiScaling();
    } else {
      Evaluate(x, acc, true);
      if (!cost_.isValid()) return false;
    }

    ApplyDamping();
#if defined(TINYOPT_ENABLE_SUITESPARSE)
    if constexpr (traits::is_sparse_matrix_v<H_t>) {
      if (this->options_.linear_solver == LinearSolverMethod::SuiteSparse) H_.makeCompressed();
    }
#endif
    UpdateTrustRegion(cost_.cost);
    ApplyLockedStates();
    return true;
  }

  bool shouldUseLockedPermutation(Index dims, Index num_locked) const {
    if constexpr (Dims != Dynamic) return false;
    if (this->options_.opt.min_dims_use_lock_permutation <= 0) return false;
    if (dims < this->options_.opt.min_dims_use_lock_permutation) return false;
    if (num_locked < this->options_.opt.min_num_locked_permutation) return false;
    return true;
  }

  void InitLockedPermutation(Index dims) {
    if constexpr (Dims != Dynamic) {
      use_locked_permutation_ = false;
      return;
    } else {
      if (!this->hasLocked()) {
        use_locked_permutation_ = false;
        return;
      }

      const Index num_locked = this->numLocked();
      if (!shouldUseLockedPermutation(dims, num_locked)) {
        use_locked_permutation_ = false;
        return;
      }

      use_locked_permutation_ = true;
      free_dims_ = dims - num_locked;

      std::vector<bool> is_locked(dims, false);
      for (auto idx : this->locked_indices_) {
        if (idx >= 0 && idx < dims) {
          is_locked[idx] = true;
        }
      }

      if constexpr (Dims == Dynamic) {
        perm_.resize(dims);
      }
      Index free_idx = 0;
      Index lock_idx = free_dims_;
      for (Index i = 0; i < dims; ++i) {
        if (is_locked[i]) {
          perm_.indices()[lock_idx++] = static_cast<int>(i);
        } else {
          perm_.indices()[free_idx++] = static_cast<int>(i);
        }
      }
    }
  }

  void ApplyLockedStates() {
    if (!this->hasLocked()) {
      return;
    }

    const Index dims = H_.rows();
    if (use_locked_permutation_) {
      if constexpr (Dims == Dynamic) {
        for (auto idx : this->locked_indices_) {
          if (idx >= 0 && idx < dims) {
            grad_(idx) = Scalar(0);
          }
        }
      } else {  // Fixed size
        for (Index idx = 0; idx < Dims; ++idx) {
          if (this->locked_indices_[idx] && idx < dims) {
            grad_(idx) = Scalar(0);
          }
        }
      }
      return;
    }

    // Sparse Matrix and Dynamic sized system
    if constexpr (traits::is_sparse_matrix_v<H_t>) {
      const auto locked_indices = this->getLockedIndices();
      std::vector<bool> is_locked(dims, false);
      for (auto idx : locked_indices) {
        if (idx >= 0 && idx < dims) {
          is_locked[idx] = true;
          grad_(idx) = Scalar(0);
        }
      }

      using Triplet = Eigen::Triplet<Scalar>;
      std::vector<Triplet> triplets;
      triplets.reserve(H_.nonZeros() + static_cast<Index>(locked_indices.size()));

      for (Index k = 0; k < H_.outerSize(); ++k) {
        for (typename H_t::InnerIterator it(H_, k); it; ++it) {
          const Index r = it.row();
          const Index c = it.col();
          if (!is_locked[r] && !is_locked[c]) {
            triplets.emplace_back(r, c, it.value());
          }
        }
      }
      for (auto idx : locked_indices) {
        if (idx >= 0 && idx < dims) {
          triplets.emplace_back(idx, idx, Scalar(1));
        }
      }
      H_.setFromTriplets(triplets.begin(), triplets.end());
    } else {  // Dense
      if constexpr (Dims == Dynamic) {
        for (auto idx : this->locked_indices_) {
          if (idx >= 0 && idx < dims) {
            H_.row(idx).setZero();
            H_.col(idx).setZero();
            H_(idx, idx) = Scalar(1);
            grad_(idx) = Scalar(0);
          }
        }
      } else {
        for (Index idx = 0; idx < Dims; ++idx) {
          if (this->locked_indices_[idx] && idx < dims) {
            H_.row(idx).setZero();
            H_.col(idx).setZero();
            H_(idx, idx) = Scalar(1);
            grad_(idx) = Scalar(0);
          }
        }
      }
    }
  }

  virtual std::optional<Grad_t> Solve() const = 0;
  virtual void GoodStep(Scalar) = 0;
  virtual void BadStep(Scalar = 0) = 0;

  void FailedStep() { BadStep(); }
  virtual void Rebuild(bool rebuild) { RebuildStrategy(rebuild); }
  virtual std::string stateAsString() const { return {}; }
  Index dims() const { return grad_.size(); }
  const Cost &cost() const { return cost_; }
  const Grad_t &Gradient() const { return grad_; }
  Grad_t &Gradient() { return grad_; }
  Scalar GradientSquaredNorm() const { return grad_.squaredNorm(); }
  const H_t &H() const { return H_; }
  H_t &H() { return H_; }
  [[nodiscard]] bool useLockedPermutation() const { return use_locked_permutation_; }

  H_t Hessian() const {
    H_t hessian = H_;
    RestoreHessian(hessian);
    return hessian;
  }

  Scalar MaxStdDev() const {
    const auto covariance = InvCov(Hessian());
    if (!covariance) return 0;
    if constexpr (traits::is_sparse_matrix_v<H_t>)
      return std::sqrt(covariance->coeffs().maxCoeff());
    else
      return std::sqrt(covariance->maxCoeff());
  }

 protected:
  static Options NormalizeOptions(const Options &options) {
    Options normalized = options;
    if (normalized.solver_type == Options::Solver::GradientDescent ||
        normalized.solver_type == Options::Solver::ConjugateGradient ||
        normalized.solver_type == Options::Solver::BFGS ||
        normalized.solver_type == Options::Solver::LBFGS)
      normalized.solver_type = Options::Solver::LevenbergMarquardt;
    return normalized;
  }

  virtual void ResetStrategy() = 0;
  virtual void ResizeStrategy(Index) {}
  virtual bool ShouldRebuildLinearSystem() const { return true; }
  virtual void ApplyJacobiScaling() {}
  virtual void ApplyDamping() {}
  virtual void UpdateTrustRegion(Scalar) {}
  virtual void RebuildStrategy(bool) {}
  virtual void RestoreHessian(H_t &) const {}

  Grad_t grad_;
  H_t H_;
  Cost cost_;
  /// Permutation matrix reordering free parameters first and locked parameters last.
  Eigen::PermutationMatrix<Dims, Dims, int> perm_;
  /// Number of free (unlocked) parameter dimensions when using permutation solving.
  Index free_dims_ = 0;
  /// Flag indicating whether permutation-based partitioned linear solve is active.
  bool use_locked_permutation_ = false;

 protected:
  template <typename VectorType>
  std::optional<Grad_t> SolveLinear(const VectorType &rhs) const {
    // For dense static sized systems, we don't do permutation
    if constexpr (Dims != Dynamic) {
      return tinyopt::SolveLinearSystem(H_, rhs, this->options_.linear_solver,
                                        this->options_.svd_relative_threshold);
    } else if (use_locked_permutation_) {
      if (free_dims_ <= 0) {
        return Grad_t::Zero(dims());
      }
      if (!this->options_.hessian.H_is_full && RequiresFullMatrix(this->options_.linear_solver)) {
        CompleteSymmetricMatrix(const_cast<H_t &>(H_));
      }
      Grad_t rhs_perm = perm_.inverse() * rhs;

      std::optional<Vector<Scalar, Dynamic>> maybe_dx_f;
      if constexpr (traits::is_sparse_matrix_v<H_t>) {
        SparseMatrix<Scalar> H_perm;
        H_perm = H_.twistedBy(perm_.inverse());
        SparseMatrix<Scalar> H_ff =
            H_perm.topLeftCorner(free_dims_, free_dims_);  // could be dense here..
        maybe_dx_f = tinyopt::SolveLinearSystem(H_ff, rhs_perm.head(free_dims_),
                                                this->options_.linear_solver,
                                                this->options_.svd_relative_threshold);
      } else {
        auto H_perm = perm_.inverse() * H_ * perm_;
        auto H_ff = H_perm.topLeftCorner(free_dims_, free_dims_);
        maybe_dx_f = tinyopt::SolveLinearSystem(H_ff, rhs_perm.head(free_dims_),
                                                this->options_.linear_solver,
                                                this->options_.svd_relative_threshold);
      }

      if (!maybe_dx_f) return std::nullopt;

      Grad_t dx_perm = Grad_t::Zero(dims());
      dx_perm.head(free_dims_) = *maybe_dx_f;
      Grad_t dx = perm_ * dx_perm;
      return dx;
    } else {
      return tinyopt::SolveLinearSystem(H_, rhs, this->options_.linear_solver,
                                        this->options_.svd_relative_threshold);
    }
  }

 private:
  void NormalizeCost(Cost &value) const {
    if (!this->options_.cost.use_squared_norm) value.cost = std::sqrt(value.cost);
    if (this->options_.cost.downscale_by_2) value.cost *= 0.5;
    if (this->options_.cost.normalize && value.num_resisuals > 0) value.cost /= value.num_resisuals;
  }
};

}  // namespace tinyopt
