// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cmath>
#include <iostream>

// Estimate a sampled sinusoid's amplitude and DC offset when its frequency is known.
// The objective is a small smooth least-squares problem suited to conjugate gradients.
#include <tinyopt/optimizers/cg.h>

// Tinyopt provides the fixed-size parameter vector and conjugate-gradient API.
using namespace tinyopt;

int main() {
  // The known signal and sample times define synthetic observations.
  constexpr double angular_frequency = 2.0 * 3.14159265358979323846;
  constexpr double true_amplitude = 1.8;
  constexpr double true_offset = 0.35;
  const std::array<double, 12> times{0.0,  0.08, 0.16, 0.24, 0.32, 0.40,
                                     0.48, 0.56, 0.64, 0.72, 0.80, 0.88};
  // The unknowns are sinusoid amplitude and constant baseline.
  Vec2 parameters = Vec2::Zero();  // Amplitude and DC offset.

  // Minimize squared sample errors; Tinyopt obtains the gradient by autodiff.
  const auto objective = [&](const auto& estimate) {
    auto error = estimate[0] * 0.0;
    for (double time : times) {
      const double basis = std::sin(angular_frequency * time);
      const auto residual =
          estimate[0] * basis + estimate[1] - (true_amplitude * basis + true_offset);
      error += residual * residual;
    }
    return error;
  };

  // Options set the iteration budget and step size for the first-order solver.
  Options options;
  options.log.enable = false;
  options.stop.max_iters = 100;
  options.stop.min_rerr_dec = 0;
  options.cg.step_size = 0.25f;
  // The optimizer updates parameters in place and returns status/iteration information.
  const auto summary = cg::Optimizer<Vec2>(options)(parameters, objective);
  std::cout << "Estimated [amplitude, offset]: " << parameters.transpose() << '\n'
            << "Optimization succeeded: " << std::boolalpha << summary.Succeeded() << '\n';
  // Propagate optimizer failure to shell scripts and calling applications.
  return summary.Succeeded() ? 0 : 1;
}