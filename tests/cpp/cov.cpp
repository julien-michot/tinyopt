// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <cmath>

#if CATCH2_VERSION == 2
#include <catch2/catch.hpp>
#else
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#endif

#include <tinyopt/tinyopt.h>

using Catch::Approx;
using namespace tinyopt;
using namespace tinyopt::losses;
using namespace tinyopt::nlls;

void TestCov() {
  // Testing with iso weights and manual gradient/hessian update
  {
    const Vec2 y = 10 * Vec2::Random();       // prior
    const Vec2 stdevs = Vec2::Constant(4.2);  // prior standard deviations

    auto loss = [&](const auto &x, auto &grad, auto &H) {
      if constexpr (!traits::is_nullptr_v<decltype(grad)>) {
        const auto &[res, J] = MahaWhitened(x - y, stdevs, true);
        grad = J * res;
        H.diagonal() = stdevs.cwiseInverse().cwiseAbs2();  // or Jt*J
        return res.norm();                                 // return √(res.t()*res)
      } else {
        const auto res = MahaWhitened(x - y, stdevs);
        return res.norm();  // return √(res.t()*res)
      }
    };

    Vec2 x(0, 0);
    Options options;
    options.log.print_J_jet = true;
    const auto &sum = Optimize(x, loss, options);
    REQUIRE(sum.Succeeded());
    REQUIRE(sum.Converged());
    REQUIRE(sum.has_final_cov());
    const Mat2 C = sum.final_cov_dense();
    REQUIRE((C.diagonal().cwiseSqrt() - stdevs).cwiseAbs().maxCoeff() == Approx(0.0).margin(1e-7));
  }
  // Testing with iso weights and AutoDiff
  {
    const Vec2 y = 10 * Vec2::Random();       // prior
    const Vec2 stdevs = Vec2::Constant(4.2);  // prior standard deviations

    auto loss = [&](const auto &x) {
      // Final error will be e = res.T * stdevs.squared().inv() * res
      return MahaWhitened(x - y, stdevs);
    };

    Vec2 x(0, 0);
    Options options;
    options.log.print_J_jet = true;
    const auto &sum = Optimize(x, loss, options);
    REQUIRE(sum.Succeeded());
    REQUIRE(sum.Converged());
    REQUIRE(sum.has_final_cov());
    const Mat2 C = sum.final_cov_dense();
    REQUIRE((C.diagonal().cwiseSqrt() - stdevs).cwiseAbs().maxCoeff() == Approx(0.0).margin(1e-5));
  }
  // Testing with a general covariance matrix and manual gradient/hessian update
  {
    const Vec2 y = 2 * Vec2::Random();  // prior
    Mat2 Cy;                            // prior covariance
    Cy << 10, 2, 2, 4;

    auto loss = [&](const auto &x, auto &grad, auto &H) {
      if constexpr (!traits::is_nullptr_v<decltype(grad)>) {
        const auto &[res, J] = MahaWhitened(x - y, Cy, true);
        grad = J * res;                  // J is stdevs.cwiseInverse().asDiagonal()
        H = J.transpose() * J;           // Jt*J
        return std::sqrt(res.dot(res));  // return √(res.t()*res)
      } else {                           // no gradient requested
        return MahaWhitened(x - y, Cy).norm();
      }
    };

    Vec2 x(0, 0);
    Options options;
    options.log.print_J_jet = true;
    const auto &sum = Optimize(x, loss, options);
    REQUIRE(sum.has_final_cov());
    const Mat2 C = sum.final_cov_dense();
    REQUIRE((C - Cy).cwiseAbs().maxCoeff() == Approx(0.0).margin(1e-5));
    REQUIRE(sum.Succeeded());
    REQUIRE(sum.Converged());
  }

  // Testing with a general covariance matrix and manual gradient/hessian update
  {
    const Vec2 y = 2 * Vec2::Random();  // prior
    Mat2 Cy;                            // prior covariance
    Cy << 10, 2, 2, 4;

    const Mat2 Lt = Cy.inverse().llt().matrixU();  // I = L*Lt
    auto loss = [&](const auto &x, auto &grad, auto &H) {
      if constexpr (!traits::is_nullptr_v<decltype(grad)>) {
        const auto &[res, J] = MahaWhitenedInfoU(x - y, Lt, true);
        grad = J * res;                  // J is stdevs.cwiseInverse().asDiagonal()
        H = J.transpose() * J;           // Jt*J
        return std::sqrt(res.dot(res));  // return √(res.t()*res)
      } else {
        return MahaWhitenedInfoU(x - y, Lt).norm();  // return √(res.t()*res)
      }
    };

    Vec2 x(0, 0);
    Options options;
    options.log.print_J_jet = true;
    const auto &sum = Optimize(x, loss, options);
    REQUIRE(sum.Succeeded());
    REQUIRE(sum.Converged());
    REQUIRE(sum.has_final_cov());
    const Mat2 C = sum.final_cov_dense();
    REQUIRE((C - Cy).cwiseAbs().maxCoeff() == Approx(0.0).margin(1e-5));
  }

  // Testing with a general covariance matrix and AD
  {
    const Vec2 y = 2 * Vec2::Random();  // prior
    Mat2 Cy;                            // prior covariance
    Cy << 10, 2, 2, 4;
    const Mat2 Lt = Cy.inverse().llt().matrixU();  // Lt

    auto loss = [&](const auto &x) {
      const auto res = Lt * (x - y);  // Final error will be e = res.T * C.inv() * res
      return res.eval();              // Don't forget the .eval() since 'res' is a glue class
    };

    Vec2 x(0, 0);
    Options options;
    options.log.print_J_jet = true;
    const auto &sum = Optimize(x, loss, options);
    REQUIRE(sum.Succeeded());
    REQUIRE(sum.Converged());
    REQUIRE(sum.has_final_cov());
    const Mat2 C = sum.final_cov_dense();
    REQUIRE((C - Cy).cwiseAbs().maxCoeff() == Approx(0.0).margin(1e-5));
  }
  // Testing with a general covariance matrix and AD
  {
    const Vec2 y = 2 * Vec2::Random();  // prior
    Mat2 Cy;                            // prior covariance
    Cy << 10, 2, 2, 4;

    auto loss = [&](const auto &x) {
      using T = typename std::decay_t<decltype(x)>::Scalar;
      const Matrix<T, 2, 2> C_ = Cy.template cast<T>();
      return MahaWhitened(x - y, C_);  // e = res.T * C.inv() * res
    };

    Vec2 x(0, 0);
    Options options;
    options.log.print_J_jet = true;
    const auto &sum = Optimize(x, loss, options);
    REQUIRE(sum.Succeeded());
    REQUIRE(sum.Converged());
    REQUIRE(sum.has_final_cov());
    const Mat2 C = sum.final_cov_dense();
    REQUIRE((C - Cy).cwiseAbs().maxCoeff() == Approx(0.0).margin(1e-5));
  }
}

TEST_CASE("tinyopt_cov") { TestCov(); }

TEST_CASE("tinyopt_inverse_matches_matrix_inverse") {
  Mat2 covariance;
  covariance << 2.0, 0.5, 0.5, 3.0;

  const auto inverse = Inverse(covariance);
  REQUIRE(inverse.has_value());
  REQUIRE((inverse->transpose() - inverse.value()).norm() < 1e-12);
  REQUIRE(((*inverse) * covariance - Mat2::Identity()).norm() < 1e-10);
}

TEST_CASE("tinyopt_inverse_dispatches_by_matrix_type_and_size") {
  Mat6 dense = Mat6::Identity() * 2.0;
  const auto dense_inverse = Inverse(dense);
  REQUIRE(dense_inverse.has_value());
  REQUIRE((*dense_inverse * dense - Mat6::Identity()).norm() < 1e-12);

  SparseMat sparse = dense.sparseView();
  const auto sparse_inverse = Inverse(sparse);
  REQUIRE(sparse_inverse.has_value());
  REQUIRE((MatX(*sparse_inverse) * MatX(sparse) - MatX::Identity(6, 6)).norm() < 1e-12);

  BlockSparseMatrix<double, 2> block_sparse(3);
  for (Index i = 0; i < 3; ++i) {
    block_sparse.block<2, 2>(i, i) = Mat2::Identity() * 2.0;
  }
  const auto block_inverse = Inverse(block_sparse);
  REQUIRE(block_inverse.has_value());
  REQUIRE((MatX(*block_inverse) * block_sparse.toDense() - MatX::Identity(6, 6)).norm() < 1e-12);
}

TEST_CASE("tinyopt_summary_exports_final_covariance") {
  double x = 0.0;
  auto residuals = [](const auto &value) {
    using T = std::decay_t<decltype(value)>;
    Vector<T, 3> result;
    result << value - T(1), value - T(2), value - T(4);
    return result;
  };

  const auto summary = Optimize(x, residuals);
  REQUIRE(summary.Succeeded());
  REQUIRE(summary.Converged());
  REQUIRE(summary.final_cost.cost == Approx(14.0 / 3.0).margin(1e-10));

  REQUIRE(summary.has_final_cov());
  REQUIRE(summary.final_cov_dense()(0, 0) == Approx(1.0 / 3.0).margin(1e-10));
}

TEST_CASE("tinyopt_summary_can_skip_final_covariance_export") {
  double x = 0.0;
  Options options;
  options.exp.final_cov = false;
  const auto summary = Optimize(x, [](const auto &value) { return value - 2.0; }, options);

  REQUIRE(summary.Succeeded());
  REQUIRE_FALSE(summary.has_final_cov());
}

TEST_CASE("tinyopt_optimizer2_exposes_covariance") {
  Vec2 x = Vec2::Zero();
  auto residuals = [](const auto &value) { return (value - Vec2(1.0, 2.0)).eval(); };
  lm::Optimizer<Mat2> optimizer;
  const auto summary = optimizer(x, residuals);

  REQUIRE(summary.Succeeded());
  const auto covariance = optimizer.Covariance();
  REQUIRE(covariance.has_value());
  REQUIRE((*covariance - Mat2::Identity()).norm() < 1e-10);
  REQUIRE((summary.final_cov_dense() - *covariance).norm() < 1e-10);
}

TEST_CASE("tinyopt_block_tridiagonal_covariance_takahashi") {
  constexpr int B = 3;
  constexpr Index N = 4;
  BlockTridiagonalMat<B> T(N);

  for (Index i = 0; i < N; ++i) {
    Matrix<double, B, B> D = Matrix<double, B, B>::Random();
    T.diagonal(i) = D * D.transpose() + Matrix<double, B, B>::Identity() * 5.0;
  }
  for (Index i = 0; i < N - 1; ++i) {
    T.off_diagonal(i) = Matrix<double, B, B>::Random();
  }

  const auto T_inv = Inverse(T);
  REQUIRE(T_inv.has_value());

  const Eigen::MatrixXd dense_inv = T.toDense().inverse();
  for (Index i = 0; i < N; ++i) {
    REQUIRE((T_inv->diagonal(i) - dense_inv.block<B, B>(i * B, i * B)).norm() < 1e-10);
  }
  for (Index i = 0; i < N - 1; ++i) {
    REQUIRE((T_inv->off_diagonal(i) - dense_inv.block<B, B>(i * B, (i + 1) * B)).norm() < 1e-10);
  }

  // Test BlockCovariance on BlockTridiagonalLinearSolvers
  BlockTridiagonalLinearSolvers<BlockTridiagonalMat<B>> solver;
  const auto cov_00 = solver.BlockCovariance(T, 0, 0);
  REQUIRE(cov_00.has_value());
  REQUIRE((*cov_00 - dense_inv.block<B, B>(0, 0)).norm() < 1e-10);

  const auto cov_02 = solver.BlockCovariance(T, 0, 2);
  REQUIRE(cov_02.has_value());
  REQUIRE((*cov_02 - dense_inv.block<B, B>(0, 2 * B)).norm() < 1e-10);
}

TEST_CASE("tinyopt_optimizer_block_covariance_and_variances") {
  using Vec6 = Vector<double, 6>;
  Vec6 x = Vec6::Zero();
  const Vec6 target = Vec6::Constant(2.0);

  auto residuals = [&](const auto &value) { return (value - target).eval(); };

  lm::Optimizer<Mat6> optimizer;
  const auto summary = optimizer(x, residuals);
  REQUIRE(summary.Succeeded());

  const auto std_devs = optimizer.BlockStandardDeviations();
  for (int i = 0; i < 6; ++i) {
    REQUIRE(std_devs[i] == Approx(1.0).margin(1e-5));
  }

  const auto diag_blocks = optimizer.BlockDiagonalCovariance();
  REQUIRE(diag_blocks.size() == 6);
  for (size_t i = 0; i < 6; ++i) {
    REQUIRE(diag_blocks[i](0, 0) == Approx(1.0).margin(1e-5));
  }

  const auto block_00 = optimizer.BlockCovariance(0, 0);
  REQUIRE(block_00.has_value());
  REQUIRE((*block_00)(0, 0) == Approx(1.0).margin(1e-5));
}

struct SchurCovTestParams {
  using Scalar = double;
  static constexpr Index Dims = 5;

  Vector<double, Dims> values = Vector<double, Dims>::Zero();
  std::pair<Index, Index> reduced() const { return {0, 2}; }

  SchurCovTestParams &operator+=(const Vector<double, Dims> &delta) {
    values += delta;
    return *this;
  }
};

TEST_CASE("tinyopt_optimizer_schur_covariance") {
  using Hessian = BlockSparseMatrix<double, 2, 3>;
  SchurCovTestParams parameters;
  parameters.values << 2.0, -1.0, 1.0, 0.5, -0.5;
  const Vector<double, 5> target = Vector<double, 5>::Zero();

  Matrix<double, 2, 2> reduced_H = 3.0 * Matrix<double, 2, 2>::Identity();
  Matrix<double, 3, 3> eliminated_H = 4.0 * Matrix<double, 3, 3>::Identity();
  Matrix<double, 2, 3> coupling = Matrix<double, 2, 3>::Zero();
  coupling(0, 0) = 0.5;
  coupling(1, 1) = -0.3;

  Matrix<double, 5, 5> dense_H = Matrix<double, 5, 5>::Zero();
  dense_H.topLeftCorner<2, 2>() = reduced_H;
  dense_H.topRightCorner<2, 3>() = coupling;
  dense_H.bottomLeftCorner<3, 2>() = coupling.transpose();
  dense_H.bottomRightCorner<3, 3>() = eliminated_H;

  Eigen::LLT<Matrix<double, 5, 5>> llt(dense_H);
  REQUIRE(llt.info() == Eigen::Success);
  const Matrix<double, 5, 5> residual_factor = llt.matrixL().toDenseMatrix().transpose();

  auto loss = [&](const SchurCovTestParams &value, Vector<double, Dynamic> &grad, Hessian &H) {
    const Vector<double, 5> diff = value.values - target;
    const Vector<double, 5> res = residual_factor * diff;
    grad = dense_H * diff;
    H.block<2, 2>(0, 0) = reduced_H;
    H.block<2, 3>(0, 0) = coupling;
    H.block<3, 3>(0, 0) = eliminated_H;
    return 0.5 * res.squaredNorm();
  };

  Options options(Options::Solver::LevenbergMarquardt);
  options.lm.damping_init = 1e-4;
  lm::Optimizer<Hessian> optimizer(options);
  const auto summary = optimizer(parameters, loss);
  REQUIRE(summary.Succeeded());
  REQUIRE(summary.Converged());

  const auto schur_cov = optimizer.SchurCovariance();

  const Matrix<double, 5, 5> dense_inv = dense_H.inverse();

  // 1. Check reduced (camera) covariance
  const Mat2 cov_reduced = schur_cov.reduced();
  REQUIRE((cov_reduced - dense_inv.topLeftCorner<2, 2>()).norm() < 1e-10);

  // 2. Check eliminated (point) marginal block
  const Matrix<double, 3, 3> cov_pt0 = schur_cov.template eliminated_block<3>(0);
  REQUIRE((cov_pt0 - dense_inv.bottomRightCorner<3, 3>()).norm() < 1e-10);

  // 3. Check eliminated blocks vector
  const auto eliminated_blocks = schur_cov.template eliminated_blocks<3>();
  REQUIRE(eliminated_blocks.size() == 1);
  REQUIRE((eliminated_blocks[0] - dense_inv.bottomRightCorner<3, 3>()).norm() < 1e-10);
}

TEST_CASE("tinyopt_optimizer_covariance_options_zero_locked_variances") {
  Vec3 x = Vec3::Zero();
  Vec3 target(1.0, 2.0, 3.0);

  auto residuals = [&](const auto &value) { return (value - target).eval(); };

  lm::Optimizer<Mat3> optimizer;
  optimizer.lock({1});

  const auto summary = optimizer(x, residuals);
  REQUIRE(summary.Succeeded());

  // Default options: zero_locked_variances = true
  CovarianceOptions cov_opts;
  cov_opts.zero_locked_variances = true;
  const auto cov = optimizer.Covariance(cov_opts);
  REQUIRE(cov.has_value());

  // Locked index 1 should have variance and cross-covariance = 0
  REQUIRE((*cov)(1, 1) == 0.0);
  REQUIRE((*cov)(0, 1) == 0.0);
  REQUIRE((*cov)(1, 0) == 0.0);
  REQUIRE((*cov)(1, 2) == 0.0);
  REQUIRE((*cov)(2, 1) == 0.0);

  // Unlocked indices should have variance = 1.0
  REQUIRE((*cov)(0, 0) == Approx(1.0).margin(1e-10));
  REQUIRE((*cov)(2, 2) == Approx(1.0).margin(1e-10));

  // If zero_locked_variances = false, the constrained identity entry gives 1.0
  cov_opts.zero_locked_variances = false;
  const auto cov_raw = optimizer.Covariance(cov_opts);
  REQUIRE(cov_raw.has_value());
  REQUIRE((*cov_raw)(1, 1) == Approx(1.0).margin(1e-4));
}

TEST_CASE("tinyopt_optimizer_covariance_options_scale_residual_variance") {
  double x = 0.0;
  auto residuals = [](const auto &value) {
    using T = std::decay_t<decltype(value)>;
    Vector<T, 3> result;
    result << value - T(2.0), value - T(3.0), value - T(1.0);
    return result;
  };

  lm::Optimizer<Mat1> optimizer;
  const auto summary = optimizer(x, residuals);
  REQUIRE(summary.Succeeded());
  REQUIRE(summary.Converged());

  CovarianceOptions opts;
  opts.scale_by_residual_variance = true;
  const auto cov_scaled = optimizer.Covariance(opts);
  REQUIRE(cov_scaled.has_value());
  REQUIRE((*cov_scaled)(0, 0) == Approx(1.0 / 3.0).margin(1e-10));
}

TEST_CASE("tinyopt_summary_covariance_dynamic_method") {
  Vec2 x = Vec2::Zero();
  auto residuals = [](const auto &value) { return (value - Vec2(1.0, 2.0)).eval(); };
  const auto summary = Optimize(x, residuals);

  REQUIRE(summary.Succeeded());
  const auto cov_dyn = summary.covariance();
  REQUIRE(cov_dyn.has_value());
  REQUIRE(cov_dyn->rows() == 2);
  REQUIRE(cov_dyn->cols() == 2);
  REQUIRE((*cov_dyn - Mat2::Identity()).norm() < 1e-10);
}