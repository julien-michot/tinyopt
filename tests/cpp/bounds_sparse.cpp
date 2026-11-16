// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <limits>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <tinyopt/tinyopt.h>

using Catch::Approx;
using namespace tinyopt;

TEST_CASE("Runtime dynamic bounds constrain sparse and structured systems", "[bounds][runtime]") {
  const VecX target = (VecX(4) << 2.0, -1.0, 3.0, 4.0).finished();
  VecX lower = VecX::Constant(4, -std::numeric_limits<double>::infinity());
  VecX upper = VecX::Constant(4, std::numeric_limits<double>::infinity());
  upper[0] = 0.5;

  auto dense_loss = [&](const auto& value, auto& gradient, auto& hessian) {
    const VecX difference = value - target;
    if constexpr (!traits::is_nullptr_v<decltype(gradient)>) {
      gradient = difference;
      hessian.setIdentity();
    }
    return Cost(0.5 * difference.squaredNorm(), 4);
  };

  SECTION("Sparse") {
    VecX x = VecX::Zero(4);
    Options options(Options::Solver::GaussNewton);
    options.opt.active_set_ratio = 0.2f;
    options.stop.max_iters = 1;
    gn::Optimizer<SparseMat> optimizer(options);
    optimizer.setBounds(lower, upper);
    const auto summary = optimizer(x, dense_loss);
    REQUIRE(summary.Succeeded());
    REQUIRE(x[0] == Approx(1.0).margin(1e-8));
    REQUIRE((x.tail(3) - target.tail(3)).norm() < 1e-8);
  }

  SECTION("Block sparse") {
    using Hessian = BlockSparseMatrix<double, 2>;
    VecX x = VecX::Zero(4);
    Options options(Options::Solver::GaussNewton);
    options.opt.active_set_ratio = 0.2f;
    options.stop.max_iters = 1;
    gn::Optimizer<Hessian> optimizer(options);
    optimizer.setBounds(lower, upper);
    const auto summary = optimizer(
        x, [&](const auto& value, auto& gradient, Hessian& hessian) {
          const VecX difference = value - target;
          if constexpr (!traits::is_nullptr_v<decltype(gradient)>) {
            gradient = difference;
            hessian.clear();
            hessian.template block<2, 2>(0, 0).setIdentity();
            hessian.template block<2, 2>(1, 1).setIdentity();
          }
          return Cost(0.5 * difference.squaredNorm(), 4);
        });
    REQUIRE(summary.Succeeded());
    REQUIRE(x[0] == Approx(1.0).margin(1e-8));
    REQUIRE((x.tail(3) - target.tail(3)).norm() < 1e-8);
  }

  SECTION("Block tridiagonal") {
    using Hessian = BlockTridiagMat<2>;
    VecX x = VecX::Zero(4);
    Options options(Options::Solver::GaussNewton);
    options.opt.active_set_ratio = 0.2f;
    options.stop.max_iters = 1;
    gn::Optimizer<Hessian> optimizer(options);
    optimizer.setBounds(lower, upper);
    const auto summary = optimizer(
        x, [&](const auto& value, auto& gradient, Hessian& hessian) {
          const VecX difference = value - target;
          if constexpr (!traits::is_nullptr_v<decltype(gradient)>) {
            gradient = difference;
            hessian.setZero();
            hessian.diagonal(0).setIdentity();
            hessian.diagonal(1).setIdentity();
          }
          return Cost(0.5 * difference.squaredNorm(), 4);
        });
    REQUIRE(summary.Succeeded());
    REQUIRE(x[0] == Approx(1.0).margin(1e-8));
    REQUIRE((x.tail(3) - target.tail(3)).norm() < 1e-8);
  }
}
