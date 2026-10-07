// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tinyopt/optimizers/optimizer2.h>

namespace tinyopt::lm {

template <typename Scalar, Index Dims>
struct LMState {
  Vector<Scalar, Dims> scaling;
  Scalar lambda = 1e-4;
  Scalar previous_lambda = 0;
  Scalar bad_factor = 2;
  bool rebuild_linear_system = true;
};

template <typename Hessian_t = MatX>
class Optimizer : public tinyopt::Optimizer2Base<Optimizer<Hessian_t>, Hessian_t> {
 public:
  using Base = tinyopt::Optimizer2Base<Optimizer<Hessian_t>, Hessian_t>;
  using Options = tinyopt::Options;
  using State = lm::LMState<typename Base::Scalar, Base::Dims>;

  explicit Optimizer(const Options &options = {})
      : Base(tinyopt::WithSolverOption(options, Options::Solver::LevenbergMarquardt,
                                       "Levenberg-Marquardt")) {
    this->reset();
  }

  std::optional<typename Base::Grad_t> Solve() const override { return SolveLM(); }

  std::optional<typename Base::Grad_t> SolveLM() const {
    if (!this->cost_.isValid()) return std::nullopt;
    auto step = this->SolveLinear(-this->grad_);
    if (step && this->options_.lm.jacobi_scaling) step->array() *= state_.scaling.array();
    return step;
  }

  void GoodStep(typename Base::Scalar quality) override {
    const auto &lm = this->options_.lm;
    typename Base::Scalar factor = lm.good_factor;
    if (quality != typename Base::Scalar(0))
      factor = std::max<typename Base::Scalar>(
          factor, typename Base::Scalar(1) -
                      std::pow(typename Base::Scalar(2) * quality - typename Base::Scalar(1), 3));
    if (state_.bad_factor != lm.bad_factor) factor /= state_.bad_factor;
    state_.previous_lambda = state_.lambda;
    state_.lambda = std::clamp<typename Base::Scalar>(
        state_.lambda * factor, lm.damping_range[0], lm.damping_range[1]);
    state_.bad_factor = lm.bad_factor;
  }

  void BadStep(typename Base::Scalar = 0) override {
    const auto &lm = this->options_.lm;
    state_.previous_lambda = state_.lambda;
    state_.lambda = std::clamp<typename Base::Scalar>(
        state_.lambda * state_.bad_factor, lm.damping_range[0], lm.damping_range[1]);
    state_.bad_factor *= lm.bad_factor;
  }

  std::string stateAsString() const override {
    std::ostringstream stream;
    stream << TINYOPT_FORMAT_NS::format("○:{:.2e} ", 1.0 / state_.lambda);
    return stream.str();
  }

 protected:
  void ResetStrategy() override {
    state_.lambda = this->options_.lm.damping_init;
    state_.previous_lambda = 0;
    state_.bad_factor = this->options_.lm.bad_factor;
    state_.rebuild_linear_system = true;
  }
  void ResizeStrategy(tinyopt::Index dims) override {
    if (this->options_.lm.jacobi_scaling) state_.scaling.resize(dims);
  }
  bool ShouldRebuildLinearSystem() const override { return state_.rebuild_linear_system; }
  void ApplyJacobiScaling() override {
    if (!this->options_.lm.jacobi_scaling) return;
    constexpr typename Base::Scalar min_diagonal = typename Base::Scalar(1e-6);
    constexpr typename Base::Scalar max_diagonal = typename Base::Scalar(1e32);
    state_.scaling.resize(this->H_.rows());
    for (tinyopt::Index index = 0; index < this->H_.rows(); ++index) {
      const auto diagonal =
          std::clamp(this->H_.coeff(index, index), min_diagonal, max_diagonal);
      state_.scaling[index] = typename Base::Scalar(1) / std::sqrt(diagonal);
    }
    this->grad_.array() *= state_.scaling.array();
    ApplyJacobiScaling(this->H_, false);
  }
  void ApplyDamping() override {
    if (state_.lambda <= 0) return;
    const auto factor =
        state_.rebuild_linear_system
            ? typename Base::Scalar(1) + state_.lambda
            : (typename Base::Scalar(1) + state_.lambda) /
                  (typename Base::Scalar(1) + state_.previous_lambda);
    for (tinyopt::Index index = 0; index < this->H_.rows(); ++index) {
      if constexpr (traits::is_sparse_matrix_v<typename Base::H_t>)
        this->H_.coeffRef(index, index) *= factor;
      else
        this->H_(index, index) *= factor;
    }
  }
  void RebuildStrategy(bool rebuild) override { state_.rebuild_linear_system = rebuild; }
  void RestoreHessian(typename Base::H_t &hessian) const override {
    if (state_.previous_lambda > 0) {
      const auto factor = typename Base::Scalar(1) + state_.previous_lambda;
      for (tinyopt::Index index = 0; index < hessian.rows(); ++index) {
        if constexpr (traits::is_sparse_matrix_v<typename Base::H_t>)
          hessian.coeffRef(index, index) /= factor;
        else
          hessian(index, index) /= factor;
      }
    }
    if (this->options_.lm.jacobi_scaling) ApplyJacobiScaling(hessian, true);
  }

 private:
  template <typename H>
  void ApplyJacobiScaling(H &hessian, bool inverse) const {
    const auto scale = [&](auto &value, tinyopt::Index row, tinyopt::Index column) {
      const auto factor = state_.scaling[row] * state_.scaling[column];
      if (inverse)
        value /= factor;
      else
        value *= factor;
    };
    if constexpr (traits::is_sparse_matrix_v<H>) {
      for (tinyopt::Index outer = 0; outer < hessian.outerSize(); ++outer)
        for (typename H::InnerIterator entry(hessian, outer); entry; ++entry)
          scale(entry.valueRef(), entry.row(), entry.col());
    } else {
      for (tinyopt::Index column = 0; column < hessian.cols(); ++column)
        for (tinyopt::Index row = 0; row < hessian.rows(); ++row)
          scale(hessian(row, column), row, column);
    }
  }

  State state_;
};

}  // namespace tinyopt::lm
