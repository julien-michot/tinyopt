// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <iomanip>
#include <iostream>
#include <memory>

#include <tinyopt/tinyopt.h>

#include "bundle_adjustment.h"
#include "bundle_adjustment_tinyopt.h"
#include "options.h"

int main() {
  using namespace tinyopt;
  using namespace tinyopt::benchmark;
  using namespace tinyopt::benchmark::bundle_adjustment;

  const TinyoptLoss loss;
  Options options = CreateOptions();
  options.stop.max_iters = 100;
  options.lm.jacobi_scaling = true;
  constexpr int runs = 10;
  double initial_cost = 0.0;
  double final_cost = 0.0;
  int iterations = 0;
  Camera final_camera;
  Point final_point;
  for (int run = 0; run < runs; ++run) {
    Problem problem = MakeProblem(20, 200);
    std::nullptr_t null_gradient{};
    SparseMat unused_hessian;
    if (run == 0) initial_cost = loss(problem, null_gradient, unused_hessian).cost;
    lm::Optimizer<SparseMat> optimizer(options);
    const auto& result = optimizer(problem, loss);
    if (!result.Succeeded() || !result.Converged()) {
      std::cerr << "Sparse bundle adjustment failed to converge on run " << run << '\n';
      return 1;
    }
    final_cost = result.final_cost.cost;
    iterations = result.num_iters;
    final_camera = problem.initial_cameras[1];
    final_point = problem.initial_points[1];
  }

  std::cout << std::setprecision(12)
            << "Problem: sparse bundle adjustment (20 cameras, 200 points)\n"
            << "Runs: " << runs << '\n'
            << "Initial cost: " << initial_cost << '\n'
            << "Final cost: " << final_cost << '\n'
            << "Iterations: " << iterations << '\n'
            << "Camera 1: " << final_camera.transpose() << '\n'
            << "Point 1: " << final_point.transpose() << '\n';
  return 0;
}