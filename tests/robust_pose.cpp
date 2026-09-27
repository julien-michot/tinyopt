// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

/// @brief Tests for robust Huber-norm pose optimization in SE(3).
///
/// Problem: Estimate a 3D pose T ∈ SE(3) from N point observations corrupted with outliers.
/// Two loss functions are compared:
///   1. Standard squared-L2  — closed form (accumulates J^T J, J^T r) via autodiff
///   2. Huber re-weighted L2 — accumulates w(r) * J^T J, w(r) * J^T r via autodiff
///      with w(r) = { 1         if ||r||² ≤ δ²  (inlier)
///                  { δ/||r||   otherwise         (outlier)
///
/// API example (from the user's perspective):
/// @code
///   // L2 loss (standard):
///   Optimize(T, [&](const auto &T_j) {
///     return T_j * p_world - p_obs;   // returns Vec3 residual
///   }, options);
///
///   // Huber robust loss: weight each observation's residual by w(r) before returning
///   Optimize(T, [&](const auto &T_j) {
///     Vec3T r = T_j * p_world - p_obs;
///     auto w = HuberWeight(r.squaredNorm(), delta * delta);
///     return (w * r).eval();           // re-weighted residual, same API
///   }, options);
/// @endcode

#include <cmath>
#include <random>
#include <vector>

#include <Eigen/Core>
#include <sophus/se3.hpp>

#if CATCH2_VERSION == 2
#include <catch2/catch.hpp>
#else
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#endif

#include <tinyopt/3rdparty/traits/sophus.h>
#include <tinyopt/diff/gradient_check.h>
#include <tinyopt/losses/robust_norms.h>
#include <tinyopt/tinyopt.h>

using Catch::Approx;
using namespace tinyopt;
using namespace tinyopt::nlls;
using namespace tinyopt::losses;

using Pose = Sophus::SE3d;

// -----------------------------------------------------------------------
// Helper: M-estimator Huber weight (scalar, double-only, no Jet)
//   w = 1           if n2 <= delta2
//   w = delta/||r|| otherwise
// This is the IRLS re-weighting factor applied to r before returning.
// -----------------------------------------------------------------------
inline double HuberWeight(double n2, double delta2) {
  if (n2 <= delta2) return 1.0;
  return std::sqrt(delta2 / std::max(n2, std::numeric_limits<double>::min()));
}

// -----------------------------------------------------------------------
// Helper: Generate synthetic 3D-point observations around a known pose
// -----------------------------------------------------------------------
struct PointObs {
  Vec3 world_pt;  ///< Ground-truth 3D point in world frame
  Vec3 measured;  ///< Measured transformed point (noisy)
  bool is_outlier;
};

inline std::vector<PointObs> GenerateObs(const Pose &T_gt, int n_inliers, int n_outliers,
                                         double noise_sigma, double outlier_scale,
                                         unsigned seed = 42) {
  std::mt19937 rng(seed);
  std::normal_distribution<double> inlier_dist(0.0, noise_sigma);
  std::normal_distribution<double> outlier_dist(0.0, outlier_scale);

  std::vector<PointObs> obs;
  obs.reserve(n_inliers + n_outliers);
  for (int i = 0; i < n_inliers; ++i) {
    Vec3 p_w = Vec3::Random() * 5.0;
    Vec3 p_m = T_gt * p_w;
    p_m += Vec3(inlier_dist(rng), inlier_dist(rng), inlier_dist(rng));
    obs.push_back({p_w, p_m, false});
  }
  for (int i = 0; i < n_outliers; ++i) {
    Vec3 p_w = Vec3::Random() * 5.0;
    Vec3 p_m = T_gt * p_w;
    p_m += Vec3(outlier_dist(rng), outlier_dist(rng), outlier_dist(rng));
    obs.push_back({p_w, p_m, true});
  }
  return obs;
}

// -----------------------------------------------------------------------
// Optimizer helpers (using Optimize's autodiff – residual-returning API)
// -----------------------------------------------------------------------

/// Optimize pose with standard L2 loss via autodiff.
/// The residuals λ = T * p_world − p_measured are returned as a stacked vector.
/// The optimizer minimizes ||residuals||² = Σ ||r_i||².
inline Pose OptimizePoseL2(Pose T, const std::vector<PointObs> &obs, const Options &options) {
  Optimize(
      T,
      [&](const auto &Tj) {
        using Scalar = typename std::decay_t<decltype(Tj)>::Scalar;
        using Vec3T = Eigen::Vector3<Scalar>;
        const int N = static_cast<int>(obs.size());
        // Fixed-size only possible at compile time; use dynamic here because N is runtime.
        // Stack residuals: r ∈ R^{3N}
        Eigen::Vector<Scalar, Eigen::Dynamic> r(N * 3);
        for (int i = 0; i < N; ++i) {
          Vec3T ri = Tj * obs[i].world_pt.cast<Scalar>() - obs[i].measured.cast<Scalar>();
          r.template segment<3>(i * 3) = ri;
        }
        return r;
      },
      options);
  return T;
}

/// Optimize pose with Huber re-weighted residuals via autodiff.
/// Each observation's residual is scaled by the M-estimator weight w(r) before accumulation.
/// The autodiff engine sees weighted residuals w * r_i, so it naturally builds:
///   grad  += (w * J_i)^T * (w * r_i) = w² J_i^T r_i       (IRLS gradient)
///   H     += (w * J_i)^T * (w * J_i) = w² J_i^T J_i        (IRLS Hessian approx)
/// Note: The Huber weight is computed on the *double* norm (not Jet) so it is treated
/// as a fixed scalar in the differentiation pass — this is the standard IRLS approximation.
inline Pose OptimizePoseHuber(Pose T, const std::vector<PointObs> &obs, double delta,
                              const Options &options) {
  const double delta2 = delta * delta;
  Optimize(
      T,
      [&](const auto &Tj) {
        using Scalar = typename std::decay_t<decltype(Tj)>::Scalar;
        using Vec3T = Eigen::Vector3<Scalar>;
        const int N = static_cast<int>(obs.size());
        Eigen::Vector<Scalar, Eigen::Dynamic> r(N * 3);
        for (int i = 0; i < N; ++i) {
          Vec3T ri = Tj * obs[i].world_pt.cast<Scalar>() - obs[i].measured.cast<Scalar>();
          // Compute weight on the real part (a) to avoid differentiating the weight itself.
          // For Jet types, ri[0].a etc. gives the scalar value; for double, ri[0] directly.
          double n2;
          if constexpr (traits::is_jet_type_v<Scalar>) {
            n2 = ri[0].a * ri[0].a + ri[1].a * ri[1].a + ri[2].a * ri[2].a;
          } else {
            n2 = static_cast<double>(ri.squaredNorm());
          }
          const Scalar w(HuberWeight(n2, delta2));
          r.template segment<3>(i * 3) = w * ri;
        }
        return r;
      },
      options);
  return T;
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

/// Sanity: L2 converges to ground truth with pure inlier observations
TEST_CASE("tinyopt_robust_l2_inliers_only", "[robust][pose][l2]") {
  // Fix seed-dependent random pose so the test is reproducible
  Vec6 xi;
  xi << 0.1, -0.2, 0.15, 0.3, -0.1, 0.2;
  const Pose T_gt = Pose::exp(xi);
  const auto obs =
      GenerateObs(T_gt, /*n_inliers=*/30, /*n_outliers=*/0, /*sigma=*/0.01, /*out=*/5.0);

  Options options;
  options.max_iters = 30;
  options.log.enable = false;

  // Slightly perturb from ground truth
  Vec6 dxi;
  dxi << 0.05, -0.05, 0.05, 0.1, -0.1, 0.05;
  const Pose T_result = OptimizePoseL2(T_gt * Pose::exp(dxi), obs, options);

  const double err = (T_gt.inverse() * T_result).log().norm();
  REQUIRE(err == Approx(0.0).margin(0.05));
}

/// Demonstrate that L2 is degraded by gross outliers (biased away from ground truth)
TEST_CASE("tinyopt_robust_l2_outliers_degrade", "[robust][pose][l2]") {
  Vec6 xi;
  xi << 0.2, -0.1, 0.3, 0.1, 0.2, -0.15;
  const Pose T_gt = Pose::exp(xi);
  // 10 outliers with 50x larger noise → L2 gets pulled badly
  const auto obs =
      GenerateObs(T_gt, /*n_inliers=*/30, /*n_outliers=*/10, /*sigma=*/0.01, /*out=*/50.0, 123);

  Options options;
  options.max_iters = 50;
  options.log.enable = false;

  Vec6 dxi;
  dxi << 0.05, -0.05, 0.04, 0.08, -0.08, 0.05;
  const Pose T_result = OptimizePoseL2(T_gt * Pose::exp(dxi), obs, options);

  // L2 with 25% gross outliers should deviate significantly from ground truth
  const double err = (T_gt.inverse() * T_result).log().norm();
  REQUIRE(err > 0.05);
}

/// Huber loss recovers a pose close to ground truth even with gross outliers
TEST_CASE("tinyopt_robust_huber_outliers_robustness", "[robust][pose][huber]") {
  Vec6 xi;
  xi << 0.2, -0.1, 0.3, 0.1, 0.2, -0.15;
  const Pose T_gt = Pose::exp(xi);
  // Same outlier setup as the L2 test above
  const auto obs =
      GenerateObs(T_gt, /*n_inliers=*/30, /*n_outliers=*/10, /*sigma=*/0.01, /*out=*/50.0, 123);

  Options options;
  options.max_iters = 50;
  options.log.enable = false;

  Vec6 dxi;
  dxi << 0.05, -0.05, 0.04, 0.08, -0.08, 0.05;
  // Huber delta = 0.3 (well above inlier noise of 0.01, well below outlier scale of 50)
  const Pose T_result = OptimizePoseHuber(T_gt * Pose::exp(dxi), obs, /*delta=*/0.3, options);

  // Huber should recover much closer to ground truth despite outliers
  const double err = (T_gt.inverse() * T_result).log().norm();
  REQUIRE(err < 0.15);
}

/// Verify: Huber gradient on a scalar n² (using autodiff check vs. numerical finite difference)
TEST_CASE("tinyopt_robust_huber_gradient_valid", "[robust][huber][gradient]") {
  // Inlier case: n² < δ² → loss = n², J_scale = 1
  {
    Vec3 r_in(0.2, -0.1, 0.15);  // ||r||² = 0.0725 < 1.0 = δ²
    Vec3 r = r_in;
    const double delta2 = 1.0;
    // gradient check: loss = HuberLoss(r, delta2) — a scalar
    bool ok = diff::CheckGradient(
        r,
        [&](const Vec3 &x, Vec3 &g) {
          const auto [loss, J] = HuberLoss(x, delta2, true);
          g.noalias() = J;
          return loss;
        },
        /*eps=*/1e-5);
    REQUIRE(ok);
  }
  // Outlier case: n² > δ² → loss = 2δ||r|| − δ², J_scale = δ/||r||
  {
    Vec3 r_out(2.0, 1.5, 1.0);  // ||r||² = 8.25 > 1.0 = δ²
    Vec3 r = r_out;
    const double delta2 = 1.0;
    bool ok = diff::CheckGradient(
        r,
        [&](const Vec3 &x, Vec3 &g) {
          const auto [loss, J] = HuberLoss(x, delta2, true);
          g.noalias() = J;
          return loss;
        },
        /*eps=*/1e-5);
    REQUIRE(ok);
  }
}
