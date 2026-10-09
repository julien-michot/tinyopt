// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <iomanip>
#include <iostream>

#include <tinyopt/tinyopt.h>

#include "../dense_loss.h"
#include "dense_problems.h"
#include "options.h"

int main() {
  using Parameters = double;
  using namespace tinyopt;
  using namespace tinyopt::benchmark;

  const DenseMathLoss loss;
  const Parameters initial = DenseMathInitial<double>(1)[0];
  Options options = CreateOptions();
  options.stop.max_iters = 100;
  double total_final_cost = 0.0;
  Parameters parameters = 0;
  Summary result;
  constexpr int runs = 100000;
  for (int run = 0; run < runs; ++run) {
    parameters = initial + 0.0001 * (run % 10);
    const auto &sum = Optimize(parameters, loss, options);
    if (!sum.Succeeded() || !sum.Converged()) {
      std::cerr << "1D optimization failed to converge on run " << run << '\n';
      return 1;
    }
    total_final_cost += result.final_cost.cost;
    if (run == runs - 1) result = sum;  // Save the last result for reporting
  }

  std::cout << std::setprecision(12) << "Problem: Tinyopt 1D nonlinear least squares\n"
            << "Runs: " << runs << '\n'
            << "Final cost: " << result.final_cost.cost << '\n'
            << "Final parameter: " << parameters << '\n'
            << "Accumulated final cost: " << total_final_cost << '\n';
  return 0;
}