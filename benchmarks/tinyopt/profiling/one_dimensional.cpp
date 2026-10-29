// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <iomanip>
#include <iostream>

#include <tinyopt/tinyopt.h>

#include "../dense_loss.h"
#include "bench.h"
#include "dense_problems.h"

int main(int argc, char** argv) {
  using namespace tinyopt;
  using namespace tinyopt::benchmark;

  const BenchConfig config = ParseBenchConfig(argc, argv);
  const DenseMathLoss loss;
  const double initial = DenseMathInitial<double>(1)[0];
  Options options = PinnedOptions();
  lm::Optimizer<Mat1> optimizer(options);

  bool converged = true;
  double parameters = initial;
  double total_final_cost = 0.0;
  double last_final_cost = 0.0;
  int last_iters = 0;

  const BenchResult bench = TimeWorkload("1d", config, [&](int run) {
    parameters = initial + 0.0001 * (run % 10);
    optimizer.reset();
    const auto sum = optimizer(parameters, loss);
    converged = converged && sum.Succeeded() && sum.Converged();
    last_final_cost = sum.final_cost.cost;
    last_iters = sum.num_iters;
    total_final_cost += last_final_cost;
    return sum.num_iters;
  });

  if (!converged) {
    std::cerr << "1D optimization failed to converge\n";
    return 1;
  }

  std::cout << std::setprecision(12) << "Problem: Tinyopt 1D nonlinear least squares\n"
            << "Runs: " << config.runs << '\n'
            << "Final cost: " << last_final_cost << '\n'
            << "Final parameter: " << parameters << '\n'
            << "Final iterations: " << last_iters << '\n'
            << "Median time: " << bench.median_us << " us/run (" << bench.ns_per_iter
            << " ns/iter)\n"
            << "Accumulated final cost: " << total_final_cost << '\n';
  return 0;
}
