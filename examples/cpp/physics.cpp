// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cmath>
#include <iostream>
#include <type_traits>

// Fit a damped harmonic oscillator's amplitude, damping, frequency, and offset
// from sampled measurements generated from a known reference signal.
#include <tinyopt/optimizers/dl.h>

// Tinyopt provides fixed-size vectors and the Dogleg trust-region optimizer.
using namespace tinyopt;

int main() {
  // Sample times and reference parameters define the synthetic measurement series.
  const std::array<double, 8> times{0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1};
  const Vec4 truth(2.5, 0.15, 1.7, 0.2);  // Amplitude, damping, frequency, offset.
  // Start close to, but not exactly at, the physical parameters to be estimated.
  Vec4 parameters(2.2, 0.10, 1.55, 0.0);

  // This is the damped-cosine measurement model y(t)=A exp(-d t) cos(w t)+c.
  const auto predict = [](const auto& p, double time) {
    using std::cos;
    using std::exp;
    return p[0] * exp(-p[1] * time) * cos(p[2] * time) + p[3];
  };
  // Return one model-minus-measurement residual per sample for nonlinear least squares.
  // Using the parameter's scalar type preserves Jet values during autodiff.
  const auto residuals = [&](const auto& p) {
    using Scalar = typename std::decay_t<decltype(p)>::Scalar;
    Eigen::Matrix<Scalar, 8, 1> result;
    for (std::size_t i = 0; i < times.size(); ++i) {
      result[static_cast<Index>(i)] = predict(p, times[i]) - predict(truth, times[i]);
    }
    return result;
  };

  // Configure the trust-region radius/failure budget for the Dogleg solve.
  Options options;
  options.log.enable = false;
  options.stop.max_iters = 300;
  options.stop.max_consec_failures = 20;
  // Mat4 is the fixed-size Hessian; Dogleg combines Gauss-Newton and steepest-descent steps.
  const auto summary = dl::Optimizer<Mat4>(options)(parameters, residuals);
  std::cout << "[amplitude, damping, frequency, offset]: " << parameters.transpose() << '\n'
            << "Optimization succeeded: " << std::boolalpha << summary.Succeeded() << '\n';
  // Make failed optimization visible to callers and automation.
  return summary.Succeeded() ? 0 : 1;
}