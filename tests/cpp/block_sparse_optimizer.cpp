// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <vector>

#include <Eigen/Eigen>

#if CATCH2_VERSION == 2
#include <catch2/catch.hpp>
#else
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#endif

#include <tinyopt/tinyopt.h>

#include <tinyopt/diff/gradient_check.h>
#include <tinyopt/diff/num_diff.h>
#include <tinyopt/optimize.h>
#include <tinyopt/optimizers/optimizer.h>

using Catch::Approx;

using namespace tinyopt;
using namespace tinyopt::diff;

namespace {

struct BlockSchurParams {
  using Scalar = double;
  static constexpr Index Dims = 5;

  Vector<double, Dims> values = Vector<double, Dims>::Zero();
  std::vector<Index> fixed_indices;

  std::pair<Index, Index> reduced() const { return {0, 2}; }
  const std::vector<Index>& fixed() const { return fixed_indices; }
  BlockSchurParams& operator+=(const Vector<double, Dims>& delta) {
    values += delta;
    return *this;
  }
};

}  // namespace

TEST_CASE("tinyopt_optimize_block_sparse_hessian") {
  Vec4 x(8.0, -6.0, 4.0, -5.0);
  using Hessian = BlockSparseMatrix<double, 2>;
  const Matrix<double, 2, 2> first_factor{{2.0, 0.3}, {0.3, 4.0}};
  const Matrix<double, 2, 2> second_factor{{3.0, 0.2}, {0.2, 5.0}};
  const Vec4 target(2.0, -1.0, 0.5, 3.0);

  Matrix<double, 4, 4> residual_factor = Matrix<double, 4, 4>::Zero();
  residual_factor.topLeftCorner<2, 2>() = first_factor;
  residual_factor.bottomRightCorner<2, 2>() = second_factor;
  const Matrix<double, 4, 4> dense_hessian = residual_factor.transpose() * residual_factor;

  auto gradient_only = [&](const Vec4& value, auto& gradient) {
    const Vec4 difference = value - target;
    const Vec4 residual = residual_factor * difference;
    if constexpr (!traits::is_nullptr_v<decltype(gradient)>) gradient = dense_hessian * difference;
    return 0.5 * residual.squaredNorm();
  };
  REQUIRE(CheckGradient(x, gradient_only, 1e-5, Method::kCentral, false));

  auto loss = [&](const Vec4& value, Vector<double, Dynamic>& gradient, Hessian& hessian) {
    const Vec4 difference = value - target;
    const Vec4 residual = residual_factor * difference;
    gradient = dense_hessian * difference;
    hessian.block<2, 2>(0, 0) = dense_hessian.topLeftCorner<2, 2>();
    hessian.block<2, 2>(1, 1) = dense_hessian.bottomRightCorner<2, 2>();
    return 0.5 * residual.squaredNorm();
  };

  Options options(Options::Solver::LevenbergMarquardt);
  options.lm.jacobi_scaling = true;
  options.lm.damping_init = 1e-3;
  options.stop.max_iters = 20;
  const auto summary = Optimize(x, loss, options);

  REQUIRE(summary.Succeeded());
  REQUIRE(summary.Converged());
  REQUIRE((x - target).norm() < 1e-6);
}

TEST_CASE("tinyopt_optimize_block_sparse_schur_hessian") {
  BlockSchurParams parameters;
  parameters.values << 5.0, -3.0, 4.0, 7.0, -2.0;
  parameters.fixed_indices = {1, 4};
  const Vector<double, 5> target{1.0, -3.0, -1.0, 0.5, -2.0};
  using Hessian = BlockSparseMatrix<double, 2, 3>;
  Matrix<double, 2, 2> reduced_hessian = 2.0 * Matrix<double, 2, 2>::Identity();
  Matrix<double, 3, 3> eliminated_hessian = 3.0 * Matrix<double, 3, 3>::Identity();
  Matrix<double, 2, 3> coupling = Matrix<double, 2, 3>::Zero();
  coupling(0, 0) = 0.25;
  coupling(1, 1) = -0.2;
  Matrix<double, 5, 5> dense_hessian = Matrix<double, 5, 5>::Zero();
  dense_hessian.topLeftCorner<2, 2>() = reduced_hessian;
  dense_hessian.topRightCorner<2, 3>() = coupling;
  dense_hessian.bottomLeftCorner<3, 2>() = coupling.transpose();
  dense_hessian.bottomRightCorner<3, 3>() = eliminated_hessian;
  Eigen::LLT<Matrix<double, 5, 5>> hessian_factorization(dense_hessian);
  REQUIRE(hessian_factorization.info() == Eigen::Success);
  const Matrix<double, 5, 5> residual_factor =
      hessian_factorization.matrixL().toDenseMatrix().transpose();

  auto loss = [&](const BlockSchurParams& value, Vector<double, Dynamic>& gradient,
                  Hessian& hessian) {
    const Vector<double, 5> difference = value.values - target;
    const Vector<double, 5> residual = residual_factor * difference;
    gradient = dense_hessian * difference;
    hessian.block<2, 2>(0, 0) = reduced_hessian;
    hessian.block<2, 3>(0, 0) = coupling;
    hessian.block<3, 3>(0, 0) = eliminated_hessian;
    return 0.5 * residual.squaredNorm();
  };

  Options options(Options::Solver::LevenbergMarquardt);
  options.lm.jacobi_scaling = true;
  options.lm.damping_init = 1e-3;
  options.exp.final_cov = true;
  lm::Optimizer<Hessian> optimizer(options);
  const auto summary = optimizer(parameters, loss);
  REQUIRE(summary.Succeeded());
  REQUIRE(summary.Converged());
  REQUIRE(parameters.values[1] == Approx(-3.0));
  REQUIRE(parameters.values[4] == Approx(-2.0));
  REQUIRE((parameters.values - target).norm() < 1e-6);
  const auto* covariance = std::get_if<BlockSparseMatX<float>>(&summary.final_cov);
  REQUIRE(covariance != nullptr);
  REQUIRE(covariance->block_sizes() == std::vector<int>{2, 3});
  REQUIRE(covariance->block_counts() == std::vector<Index>{1, 1});
  REQUIRE((covariance->toDense().template cast<double>() - optimizer.Hessian().toDense().inverse())
              .norm() < 1e-6);
}
