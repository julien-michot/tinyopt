// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <iomanip>
#include <iostream>

#include <tinyopt/tinyopt.h>

#include "bench.h"

using namespace tinyopt;

/// Cyclic 6D nonlinear least squares:
///   r_i = x_i^2 + x_{i+1} - 2,  i = 0..5 (indices modulo 6)
/// whose unique nearby solution is x = 1. With 6 parameters this exercises the
/// dense linear solver path that is not covered by the 1D/2D small-system
/// closed-form solvers (dimensions greater than four).
struct SixDimLoss {
  template <typename Parameters, typename Gradient, typename Hessian>
  tinyopt::Cost operator()(const Parameters& x, Gradient& gradient, Hessian& hessian) const {
    using Scalar = typename Parameters::Scalar;
    constexpr int Dims = 6;
    Vector<Scalar, Dims> residuals;
    if constexpr (!tinyopt::traits::is_nullptr_v<Gradient>) {  // With gradient+hessian
      Matrix<Scalar, Dims, Dims> jacobian = Matrix<Scalar, Dims, Dims>::Zero();
      for (int i = 0; i < Dims; ++i) {
        const int next = (i + 1) % Dims;
        residuals[i] = x[i] * x[i] + x[next] - Scalar(2);
        jacobian(i, i) = Scalar(2) * x[i];
        jacobian(i, next) = Scalar(1);
      }
      gradient = jacobian.transpose() * residuals;
      hessian = jacobian.transpose() * jacobian;
    } else {  // Residuals only
      for (int i = 0; i < Dims; ++i) {
        const int next = (i + 1) % Dims;
        residuals[i] = x[i] * x[i] + x[next] - Scalar(2);
      }
    }
    return tinyopt::Cost(Scalar(0.5) * residuals.squaredNorm(), Dims);
  }
};

int main(int argc, char** argv) {
  using Parameters = Eigen::Matrix<double, 6, 1>;
  using Hessian = Eigen::Matrix<double, 6, 6>;
  using namespace tinyopt;
  using namespace tinyopt::benchmark;

  const BenchConfig config = ParseBenchConfig(argc, argv);
  const SixDimLoss loss;
  const Parameters initial = Parameters::Constant(1.5);
  Options options = PinnedOptions();
  lm::Optimizer<Hessian> optimizer(options);

  bool converged = true;
  Parameters parameters = Parameters::Zero();
  double total_final_cost = 0.0;
  double last_final_cost = 0.0;
  int last_iters = 0;

  const BenchResult bench = TimeWorkload("6d", config, [&](int run) {
    parameters = initial;
    parameters[0] += 0.0001 * (run % 10);
    optimizer.reset();
    const auto sum = optimizer(parameters, loss);
    converged = converged && sum.Succeeded() && sum.Converged();
    last_final_cost = sum.final_cost.cost;
    last_iters = sum.num_iters;
    total_final_cost += last_final_cost;
    return sum.num_iters;
  });

  if (!converged) {
    std::cerr << "6D optimization failed to converge\n";
    return 1;
  }

  std::cout << std::setprecision(12) << "Problem: Tinyopt 6D cyclic nonlinear least squares\n"
            << "Runs: " << config.runs << '\n'
            << "Final cost: " << last_final_cost << '\n'
            << "Final parameters: " << parameters.transpose() << '\n'
            << "Final iterations: " << last_iters << '\n'
            << "Median time: " << bench.median_us << " us/run (" << bench.ns_per_iter
            << " ns/iter)\n"
            << "Accumulated final cost: " << total_final_cost << '\n';
  return 0;
}
