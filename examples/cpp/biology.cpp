// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cmath>
#include <iostream>

// Fit a logistic population-growth curve. The optimized variables are log(K)
// and log(r), so exponentiation guarantees positive carrying capacity and rate.
#include <tinyopt/optimizers/bfgs.h>

// Tinyopt provides the fixed-size parameter vector and BFGS optimizer API.
using namespace tinyopt;

int main() {
  // Synthetic population observations at known times; the model starts at N(0).
  constexpr double initial_population = 80.0;
  const std::array<double, 6> times{0.0, 2.0, 4.0, 6.0, 8.0, 10.0};
  const std::array<double, 6> population{80.0, 179.3, 362.1, 618.2, 868.2, 1038.5};
  // Initialize log parameters near plausible biological values (K=900, r=0.35).
  Vec2 log_parameters(std::log(900.0), std::log(0.35));

  // Minimize the normalized sum of squared differences between logistic predictions and data.
  // Generic arithmetic allows Tinyopt's automatic differentiation to compute the gradient.
  const auto objective = [&](const auto& parameters) {
    using std::exp;
    const auto carrying_capacity = exp(parameters[0]);
    const auto growth_rate = exp(parameters[1]);
    auto error = parameters[0] * 0.0;
    for (std::size_t i = 0; i < times.size(); ++i) {
      const auto prediction =
          carrying_capacity /
          (1.0 + (carrying_capacity / initial_population - 1.0) * exp(-growth_rate * times[i]));
      const auto residual = (prediction - population[i]) / 1000.0;
      error += residual * residual;
    }
    return error;
  };

  // Configure the quasi-Newton iteration; a small bounded step helps this nonlinear fit.
  Options options;
  options.log.enable = false;
  options.stop.max_iters = 1500;
  options.stop.min_rerr_dec = 0;
  options.bfgs.step_size = 0.05f;
  options.bfgs.max_step_size = 0.05f;

  // BFGS estimates curvature from successive gradients; the parameter vector is updated in place.
  const auto summary = bfgs::Optimizer<Vec2>(options)(log_parameters, objective);
  // Parameters are stored in log space during the solve; exponentiate them for interpretation.
  std::cout << "Estimated carrying capacity: " << std::exp(log_parameters[0]) << '\n'
            << "Estimated growth rate: " << std::exp(log_parameters[1]) << " per time unit\n"
            << "Optimization succeeded: " << std::boolalpha << summary.Succeeded() << '\n';
  // Report optimizer failure to the shell through the executable's exit status.
  return summary.Succeeded() ? 0 : 1;
}