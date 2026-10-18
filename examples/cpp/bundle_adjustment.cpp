// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

// Bundle adjustment with camera poses reduced via Schur complement and anchor camera locked.
#include <tinyopt/optimizers/gn.h>
#include <tinyopt/optimizers/lm.h>

using namespace tinyopt;

// A simple structure-from-motion state: 2 cameras (3D translations) and 3 landmarks (3D points).
// Total parameters: 2 * 3 + 3 * 3 = 15 dimensions.
// - Head 6 dimensions: Camera 0 (indices 0..2) and Camera 1 (indices 3..5).
// - Tail 9 dimensions: Landmark 0 (6..8), Landmark 1 (9..11), Landmark 2 (12..14).
struct SceneParameters {
  using Scalar = double;
  static constexpr Index Dims = 15;

  Vector<double, 15> values;

  SceneParameters() { values.setZero(); }
  explicit SceneParameters(const Vector<double, 15> &v) : values(v) {}

  // reduced() trait: the first 6 parameters (cameras) form the reduced head block for Schur
  // complement
  std::pair<Index, Index> reduced() const { return {0, 6}; }

  // locked() trait: Camera 0 is anchored at the origin (indices 0, 1, 2 locked) to fix gauge
  // ambiguity
  std::vector<Index> locked() const { return {0, 1, 2}; }

  template <typename T2>
  SceneParameters cast() const {
    SceneParameters p;
    p.values = values.template cast<double>();
    return p;
  }

  template <typename DeltaType>
  SceneParameters &operator+=(const DeltaType &delta) {
    values += delta;
    return *this;
  }
};

int main() {
  std::cout << "=== Tinyopt Example: Bundle Adjustment with Schur Complement & Locked Anchor ===\n";

  // Ground truth setup:
  // Camera 0 at origin [0, 0, 0] (fixed gauge anchor)
  // Camera 1 translated to [1.5, 0.0, 0.0]
  // Landmarks at [0.0, 0.0, 5.0], [1.0, 1.0, 6.0], [-1.0, 0.5, 4.0]
  Vector<double, 15> ground_truth;
  ground_truth << 0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 5.0, 1.0, 1.0, 6.0, -1.0, 0.5, 4.0;

  // Initial perturbed estimate:
  SceneParameters state;
  state.values = ground_truth;
  // Perturb Camera 1 and all landmarks (Camera 0 stays at 0)
  state.values.segment<3>(3) += Vec3(0.3, -0.2, 0.1);
  state.values.segment<3>(6) += Vec3(0.2, 0.1, -0.3);
  state.values.segment<3>(9) += Vec3(-0.2, 0.3, 0.1);
  state.values.segment<3>(12) += Vec3(0.1, -0.2, 0.2);

  std::cout << "Initial Camera 1 position: " << state.values.segment<3>(3).transpose() << '\n';

  // Quadratic reprojection loss with sparse Hessian accumulation
  auto loss = [&](const auto &param, auto &grad, SparseMat &H) {
    const Vector<double, 15> diff = param.values - ground_truth;

    if constexpr (!traits::is_nullptr_v<decltype(grad)>) {
      std::vector<Eigen::Triplet<double>> triplets;

      // Diagonal blocks: 4.0 for cameras, 2.0 for landmarks
      for (Index i = 0; i < 6; ++i) triplets.emplace_back(i, i, 4.0);
      for (Index i = 6; i < 15; ++i) triplets.emplace_back(i, i, 2.0);

      // Camera-landmark coupling blocks (between cameras 0..5 and landmarks 6..14)
      triplets.emplace_back(3, 6, 0.4);
      triplets.emplace_back(6, 3, 0.4);
      triplets.emplace_back(4, 7, 0.3);
      triplets.emplace_back(7, 4, 0.3);
      triplets.emplace_back(5, 11, 0.5);
      triplets.emplace_back(11, 5, 0.5);

      H.resize(15, 15);
      H.setFromTriplets(triplets.begin(), triplets.end());
      grad = H * diff;
    }

    return 0.5 * diff.squaredNorm();
  };

  Options options(Options::Solver::GaussNewton);
  options.linear.method = LinearSolverMethod::LDLT;
  options.log.enable = true;

  gn::Optimizer<SparseMat> opt(options);
  const auto summary = opt.Optimize(state, loss);

  std::cout << "\nOptimization succeeded: " << std::boolalpha << summary.Succeeded() << '\n'
            << "Optimization converged: " << summary.Converged() << '\n'
            << "Iterations: " << summary.num_iters << '\n'
            << "Schur complement active: " << std::boolalpha << opt.hasReduced() << '\n'
            << "Reduced parameters (cameras): " << opt.numReduced() << '\n'
            << "Final Camera 0 (locked): " << state.values.segment<3>(0).transpose() << '\n'
            << "Final Camera 1 (fitted): " << state.values.segment<3>(3).transpose() << '\n'
            << "Camera 1 estimation error: "
            << (state.values.segment<3>(3) - ground_truth.segment<3>(3)).norm() << '\n';

  return summary.Succeeded() ? 0 : 1;
}
