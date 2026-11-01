// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <tinyopt/losses/robust_norms.h>
#include <tinyopt/tinyopt.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using Catch::Approx;
using namespace tinyopt;

TEST_CASE("tinyopt_tutorial_losses") {
  Vec2 robust_x = Vec2::Zero();
  auto robust_summary = tinyopt::Optimize(robust_x, [&](const auto &value) {
    const auto residual = (value - Vec2(1.0, 1.0)).eval();
    return losses::Huber(residual.squaredNorm(), 0.7);
  });
  REQUIRE(robust_summary.Succeeded());

  VecX sparse_x = VecX::Constant(5, 1.0);
  const VecX sparse_target = VecX::Constant(5, 3.0);
  auto sparse_loss = [&](auto &x, auto &grad, SparseMat &hessian) {
    const VecX res = x - sparse_target;
    if constexpr (!traits::is_nullptr_v<decltype(grad)>) {
      grad = res;
      std::vector<Eigen::Triplet<double>> triplets;
      triplets.reserve(x.size());
      for (int i = 0; i < x.size(); ++i) {
        triplets.emplace_back(i, i, 1.0);
      }
      hessian.setFromTriplets(triplets.begin(), triplets.end());
    }
    return 0.5 * res.squaredNorm();
  };
  auto sparse_summary = tinyopt::Optimize(sparse_x, sparse_loss);
  REQUIRE(sparse_summary.Succeeded());
  REQUIRE(sparse_summary.Converged());
  REQUIRE((sparse_x - sparse_target).norm() == Approx(0.0).margin(1e-5));

  using BlockHessian = BlockSparseMatrix<double, 2>;
  Vec4 block_sparse_x(8.0, -6.0, 4.0, -5.0);
  const Vec4 block_sparse_target(2.0, -1.0, 0.5, 3.0);
  const Mat2 first_block{{2.0, 0.3}, {0.3, 4.0}};
  const Mat2 second_block{{3.0, 0.2}, {0.2, 5.0}};
  auto block_sparse_loss = [&](const Vec4 &value, VecX &gradient, BlockHessian &hessian) {
    const Vec4 difference = value - block_sparse_target;
    gradient.head<2>().noalias() = first_block * difference.head<2>();
    gradient.tail<2>().noalias() = second_block * difference.tail<2>();
    hessian.block<2, 2>(0, 0) = first_block;
    hessian.block<2, 2>(1, 1) = second_block;
    return 0.5 * difference.squaredNorm();
  };
  auto block_sparse_summary = tinyopt::Optimize(block_sparse_x, block_sparse_loss);
  REQUIRE(block_sparse_summary.Succeeded());
  REQUIRE(block_sparse_summary.Converged());
  REQUIRE((block_sparse_x - block_sparse_target).norm() == Approx(0.0).margin(1e-5));
}
