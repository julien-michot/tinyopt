// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <iostream>
#include <type_traits>

// Recover one-dimensional astrometry parameters from simulated observations:
// reference position, proper motion, and annual parallax.
#include <tinyopt/optimizers/lm.h>

// Tinyopt provides the parameter types, solver options, and LM optimizer below.
using namespace tinyopt;

int main() {
  // Each epoch has a known parallax factor and a measured apparent position.
  const std::array<double, 4> epochs{-1.5, -0.5, 0.5, 1.5};
  const std::array<double, 4> parallax_factors{1.0, 0.0, -1.0, 0.0};
  const std::array<double, 4> measured_positions{0.075, 0.085, 0.095, 0.145};
  // The unknowns start at zero and are ordered as position, proper motion, parallax.
  Vec3 astrometry = Vec3::Zero();  // Reference position, proper motion, parallax.

  // Each residual compares the linear-motion-plus-parallax model with one observation.
  // A fixed-size residual vector enables Tinyopt's automatic differentiation path.
  const auto residuals = [&](const auto& parameters) {
    using Scalar = typename std::decay_t<decltype(parameters)>::Scalar;
    Eigen::Matrix<Scalar, 4, 1> result;
    for (std::size_t i = 0; i < epochs.size(); ++i) {
      result[static_cast<Index>(i)] = parameters[0] + parameters[1] * epochs[i] +
                                      parameters[2] * parallax_factors[i] - measured_positions[i];
    }
    return result;
  };

  // LM adds adaptive damping to the least-squares step, useful when parameters are correlated.
  Options options;
  options.log.enable = false;
  // Mat3 stores the 3-by-3 local Hessian; the summary reports convergence/success details.
  const auto summary = lm::Optimizer<Mat3>(options)(astrometry, residuals);
  std::cout << "[position, proper motion, parallax]: " << astrometry.transpose() << '\n'
            << "Optimization succeeded: " << std::boolalpha << summary.Succeeded() << '\n';
  // Return nonzero if the optimization did not succeed.
  return summary.Succeeded() ? 0 : 1;
}