// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

/// @brief Benchmark: Robust Huber vs. Standard L2 pose optimization.
///
/// Measures wall-clock time for Levenberg-Marquardt pose optimization in SE(3)
/// with N fixed 3D points (25% outliers) under two loss functions:
///   • L2  : minimize Σ ||T p_i − m_i||²
///   • Huber: minimize Σ ρ_H( ||T p_i − m_i|| ) using IRLS re-weighting

#include <cmath>
#include <random>
#include <vector>

#include <Eigen/Core>
#include <sophus/se3.hpp>

#if CATCH2_VERSION == 2
#include <catch2/catch.hpp>
#else
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#endif

#include <tinyopt/3rdparty/traits/sophus.h>
#include <tinyopt/losses/robust_norms.h>
#include <tinyopt/tinyopt.h>

#include "options.h"
#include "utils.h"

using namespace tinyopt;
using namespace tinyopt::nlls;
using namespace tinyopt::losses;
using namespace tinyopt::benchmark;

using Pose = Sophus::SE3d;

// -----------------------------------------------------------------------
// Shared observation data (generated once, reused across benchmarks)
// -----------------------------------------------------------------------
struct BenchObservation {
  Vec3 world_pt;
  Vec3 measured;
};

/// Build a fixed observation set: n_inliers + n_outliers points around T_gt.
static std::vector<BenchObservation> MakeBenchObs(const Pose &T_gt, int n_inliers, int n_outliers,
                                                  double sigma_in, double sigma_out,
                                                  unsigned seed = 42) {
  std::mt19937 rng(seed);
  std::normal_distribution<double> d_in(0.0, sigma_in);
  std::normal_distribution<double> d_out(0.0, sigma_out);

  std::vector<BenchObservation> obs;
  obs.reserve(n_inliers + n_outliers);

  // Reproducible points: use seeded random rather than Eigen::Random()
  std::uniform_real_distribution<double> coord(-5.0, 5.0);
  for (int i = 0; i < n_inliers; ++i) {
    Vec3 p(coord(rng), coord(rng), coord(rng));
    Vec3 m = T_gt * p + Vec3(d_in(rng), d_in(rng), d_in(rng));
    obs.push_back({p, m});
  }
  for (int i = 0; i < n_outliers; ++i) {
    Vec3 p(coord(rng), coord(rng), coord(rng));
    Vec3 m = T_gt * p + Vec3(d_out(rng), d_out(rng), d_out(rng));
    obs.push_back({p, m});
  }
  return obs;
}

// -----------------------------------------------------------------------
// Huber weight helper (applied on double scalar — constant w.r.t. Jet)
// -----------------------------------------------------------------------
inline double HuberWeight(double n2, double delta2) {
  if (n2 <= delta2) return 1.0;
  return std::sqrt(delta2 / std::max(n2, std::numeric_limits<double>::min()));
}

// -----------------------------------------------------------------------
// Benchmark: L2 vs Huber for different observation counts
// -----------------------------------------------------------------------
TEMPLATE_TEST_CASE("PoseOptimization", "[benchmark][robust][pose]", Vec6) {
  // Ground-truth pose (small rotation + translation)
  Vec6 xi;
  xi << 0.15, -0.1, 0.2, 0.3, -0.2, 0.1;
  const Pose T_gt = Pose::exp(xi);

  // Small perturbation applied at the start of every benchmark run
  Vec6 dxi;
  dxi << 0.05, -0.04, 0.06, 0.07, -0.06, 0.04;
  const Pose T_perturb = Pose::exp(dxi);

  auto n_obs = GENERATE(20, 50, 100);
  const int n_out = n_obs / 4;  // 25 % outliers
  const int n_in = n_obs - n_out;

  CAPTURE(n_obs, n_out);

  // Generate observations once
  const auto obs = MakeBenchObs(T_gt, n_in, n_out, /*sigma_in=*/0.01, /*sigma_out=*/50.0);
  const double delta2 = 0.3 * 0.3;  // Huber threshold²

  // Options: fixed iteration budget for fair comparison
  Options options = CreateOptions(/*enable_log=*/false);
  options.max_iters = 10;

  // --- L2 benchmark ---
  static StatCounter<Vec6> cnt_l2;
  BENCHMARK("L2") {
    Pose T = T_gt * T_perturb;
    const auto &out = Optimize(
        T,
        [&](const auto &Tj) {
          using Scalar = typename std::decay_t<decltype(Tj)>::Scalar;
          using Vec3T = Eigen::Vector3<Scalar>;
          Eigen::Vector<Scalar, Eigen::Dynamic> r(n_obs * 3);
          for (int i = 0; i < n_obs; ++i) {
            Vec3T ri = Tj * obs[i].world_pt.cast<Scalar>() - obs[i].measured.cast<Scalar>();
            r.template segment<3>(i * 3) = ri;
          }
          return r;
        },
        options);
    cnt_l2.AddConv(out.Converged());
    cnt_l2.AddFinalIters(out.num_iters);
    return out.error;
  };

  // --- Huber benchmark ---
  static StatCounter<Vec6> cnt_h;
  BENCHMARK("Huber") {
    Pose T = T_gt * T_perturb;
    const auto &out = Optimize(
        T,
        [&](const auto &Tj) {
          using Scalar = typename std::decay_t<decltype(Tj)>::Scalar;
          using Vec3T = Eigen::Vector3<Scalar>;
          Eigen::Vector<Scalar, Eigen::Dynamic> r(n_obs * 3);
          for (int i = 0; i < n_obs; ++i) {
            Vec3T ri = Tj * obs[i].world_pt.cast<Scalar>() - obs[i].measured.cast<Scalar>();
            // Compute weight on scalar part only (constant w.r.t. Jet)
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
    cnt_h.AddConv(out.Converged());
    cnt_h.AddFinalIters(out.num_iters);
    return out.error;
  };
}
