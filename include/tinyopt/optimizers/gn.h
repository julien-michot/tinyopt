// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tinyopt/optimizers/optimizer2.h>

namespace tinyopt::gn {

template <typename Hessian_t = MatX>
class Optimizer : public tinyopt::Optimizer2Base<Optimizer<Hessian_t>, Hessian_t> {
 public:
  using Base = tinyopt::Optimizer2Base<Optimizer<Hessian_t>, Hessian_t>;
  static constexpr tinyopt::Options::Solver Type = tinyopt::Options::Solver::GaussNewton;

  struct Options : tinyopt::Options {
    Options() : tinyopt::Options(Type) {}
    Options(const tinyopt::Options &base_options) : tinyopt::Options(base_options) {
      if (base_options.solver_type != Type) {
        TINYOPT_LOG("⚠ Optimizer received a different solver type, using GaussNewton");
      }
    }
  };

  Optimizer()
      : Optimizer([]() -> const Options & {
          static const Options default_opts(Type);
          return default_opts;
        }()) {}

  explicit Optimizer(const Options &options) : Base(options) { this->reset(); }

  std::optional<typename Base::Grad_t> Solve() const override { return SolveGN(); }
  std::optional<typename Base::Grad_t> SolveGN() const {
    if (!this->cost_.isValid()) return std::nullopt;
    return this->SolveLinear(this->grad_);
  }

 protected:
  void ResetStrategy() override {}

 public:
  void GoodStep(typename Base::Scalar) override {}
  void BadStep(typename Base::Scalar = 0) override {}
};

}  // namespace tinyopt::gn
