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