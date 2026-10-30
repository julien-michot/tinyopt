// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <iomanip>
#include <iostream>

#include <tinyopt/tinyopt.h>

#include "../dense_loss.h"
#include "bench.h"
#include "dense_problems.h"

int main(int argc, char** argv) {
  using Parameters = Eigen::Vector2d;
  using Hessian = Eigen::Matrix2d;
  using namespace tinyopt;
  using namespace tinyopt::benchmark;

  const BenchConfig config = ParseBenchConfig(argc, argv);
  const DenseMathLoss loss;
  const Parameters initial = DenseMathInitial<double>(2);
  Options options = PinnedOptions();
  lm::Optimizer<Hessian> optimizer(options);

  bool converged = true;
  Parameters parameters = Parameters::Zero();
  double total_final_cost = 0.0;
  double last_final_cost = 0.0;
  int last_iters = 0;

  const BenchResult bench = TimeWorkload("2d", config, [&](int run) {
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
    std::cerr << "2D optimization failed to converge\n";
    return 1;
  }

  std::cout << std::setprecision(12) << "Problem: Tinyopt 2D nonlinear least squares\n"
            << "Runs: " << config.runs << '\n'
            << "Final cost: " << last_final_cost << '\n'
            << "Final parameters: " << parameters.transpose() << '\n'
            << "Final iterations: " << last_iters << '\n'
            << "Median time: " << bench.median_us << " us/run (" << bench.ns_per_iter
            << " ns/iter)\n"
            << "Accumulated final cost: " << total_final_cost << '\n';
  return 0;
}
