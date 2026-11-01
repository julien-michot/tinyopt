// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <tinyopt/tinyopt.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "tutorial_common.h"

using Catch::Approx;
using namespace tinyopt;

TEST_CASE("tinyopt_tutorial_fixed_schur_covariance") {
  // Fixed parameters tutorial example
  TutorialFixedParams<double> fixed_params;
  const Vec3 target_3d{1.0, 2.0, 3.0};
  auto fixed_params_summary = tinyopt::Optimize(fixed_params, [&](const auto &p) {
    return (p.values - target_3d.template cast<typename std::decay_t<decltype(p)>::Scalar>())
        .eval();
  });
  REQUIRE(fixed_params_summary.Succeeded());
  REQUIRE(fixed_params.values[0] == Approx(target_3d[0]).margin(1e-5));
  REQUIRE(fixed_params.values[1] == Approx(5.0));  // Remains unchanged
  REQUIRE(fixed_params.values[2] == Approx(target_3d[2]).margin(1e-5));

  // 1D scalar fixed tutorial example
  float scalar_fixed_x = 3.0f;
  gn::Optimizer<Matrix<float, 1, 1>> scalar_fixed_opt;
  scalar_fixed_opt.fix({0});
  auto scalar_fixed_summary =
      scalar_fixed_opt.Optimize(scalar_fixed_x, [](const auto &val, auto &g, auto &H) {
        const float diff = val - 10.0f;
        if constexpr (!traits::is_nullptr_v<decltype(g)>) {
          g(0) = diff;
          H(0, 0) = 1.0f;
        }
        return 0.5f * diff * diff;
      });
  REQUIRE(scalar_fixed_summary.Succeeded());
  REQUIRE(scalar_fixed_x == Approx(3.0f));

  // Sparse Schur complement tutorial example
  TutorialSchurParams schur_p;
  const Vector<double, 4> schur_target{1.0, 2.0, 3.0, 4.0};
  auto schur_loss = [&](const auto &p, auto &grad, SparseMat &hessian) {
    const Vector<double, 4> diff = p.values - schur_target;
    if constexpr (!traits::is_nullptr_v<decltype(grad)>) {
      std::vector<Eigen::Triplet<double>> triplets;
      for (int i = 0; i < 4; ++i) triplets.emplace_back(i, i, 2.0);
      triplets.emplace_back(0, 2, 0.5);
      triplets.emplace_back(2, 0, 0.5);
      hessian.resize(4, 4);
      hessian.setFromTriplets(triplets.begin(), triplets.end());
      grad = hessian * diff;
    }
    return 0.5 * diff.squaredNorm();
  };
  gn::Optimizer<SparseMat> schur_opt;
  auto schur_summary = schur_opt.Optimize(schur_p, schur_loss);
  REQUIRE(schur_summary.Succeeded());
  REQUIRE(schur_opt.hasReduced());
  REQUIRE(schur_opt.numReduced() == 2);
  REQUIRE((schur_p.values - schur_target).norm() == Approx(0.0).margin(1e-5));

  // Covariance tutorial example
  Vec2 cov_x = Vec2::Zero();
  auto cov_res = [](const auto &val) { return (val - Vec2(1.0, 2.0)).eval(); };
  lm::Optimizer<Mat2> cov_opt;
  const auto cov_summary = cov_opt(cov_x, cov_res);
  REQUIRE(cov_summary.Succeeded());
  const auto cov = cov_opt.Covariance();
  REQUIRE(cov.has_value());
  REQUIRE((*cov - Mat2::Identity()).norm() < 1e-6);
  const auto std_devs = cov_opt.BlockStandardDeviations();
  REQUIRE(std_devs[0] == Approx(1.0).margin(1e-5));

  // Final covariance tutorial example
  cov_x = Vec2::Zero();
  Options cov_options;
  cov_options.exp.final_cov = true;
  const auto final_cov_summary = Optimize(cov_x, cov_res, cov_options);
  REQUIRE(final_cov_summary.has_final_cov());
  REQUIRE((std::get<MatXf>(final_cov_summary.final_cov).template cast<double>() - Mat2::Identity())
              .norm() < 1e-6);
}
