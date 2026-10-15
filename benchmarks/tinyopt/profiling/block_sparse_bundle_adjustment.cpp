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

  const BlockSparseLoss loss;
  Options options = CreateOptions();
  options.stop.max_iters = 100;
  options.lm.jacobi_scaling = true;
  options.log.enable = false;
  constexpr int runs = 1;
  double initial_cost = 0.0;
  double final_cost = 0.0;
  int iterations = 0;
  Camera final_camera;
  Point final_point;
  for (int run = 0; run < runs; ++run) {
    std::cout << "Creating problem 10c 3000p...\n" << std::flush;
    Problem problem = MakeProblem(10, 3000);
    std::cout << "Evaluating initial cost...\n" << std::flush;
    VecX gradient = VecX::Zero(6 * problem.CameraCount() + 3 * problem.PointCount());
    BlockSparseLoss::Hessian hessian(problem.CameraCount(), problem.PointCount());

    // Benchmark 1: Loss evaluation (residuals, Jacobians, and BlockSparse accumulation)
    const auto t_loss_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; ++i) {
      gradient.setZero();
      hessian.setZero();
      loss(problem, gradient, hessian);
    }
    const auto t_loss_end = std::chrono::high_resolution_clock::now();
    const double loss_ms =
        std::chrono::duration<double, std::milli>(t_loss_end - t_loss_start).count() / 10.0;
    std::cout << "1 evaluation of loss(problem, grad, hessian): " << loss_ms << " ms\n"
              << std::flush;

    // Benchmark 2: Schur solver
    tinyopt::BlockSparseLinearSolvers<BlockSparseLoss::Hessian> solver;
    solver.Analyze(hessian, 6 * problem.CameraCount());

    const auto t_compute_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; ++i) {
      solver.Compute(hessian);
    }
    const auto t_compute_end = std::chrono::high_resolution_clock::now();
    const double compute_ms =
        std::chrono::duration<double, std::milli>(t_compute_end - t_compute_start).count() / 10.0;
    std::cout << "1 call to solver.Compute(hessian): " << compute_ms << " ms\n" << std::flush;

    const auto t_solve_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; ++i) {
      solver.Solve(hessian, gradient);
    }
    const auto t_solve_end = std::chrono::high_resolution_clock::now();
    const double solve_ms =
        std::chrono::duration<double, std::milli>(t_solve_end - t_solve_start).count() / 10.0;
    std::cout << "1 call to solver.Solve(hessian, grad): " << solve_ms << " ms\n" << std::flush;
    lm::Optimizer<BlockSparseLoss::Hessian> optimizer(options);
    const auto start_time = std::chrono::high_resolution_clock::now();
    const auto& result = optimizer(problem, loss);
    const auto end_time = std::chrono::high_resolution_clock::now();
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(end_time - start_time).count();
    std::cout << "Elapsed time: " << elapsed_ms << " ms, iters: " << result.num_iters
              << ", final_cost: " << result.final_cost.cost
              << ", stop_reason: " << static_cast<int>(result.stop_reason) << '\n'
              << std::flush;
    if (!result.Succeeded() || !result.Converged()) {
      std::cerr << "Block-sparse bundle adjustment failed to converge on run " << run << '\n';
      return 1;
    }
    final_cost = result.final_cost.cost;
    iterations = result.num_iters;
    final_camera = problem.initial_cameras[1];
    final_point = problem.initial_points[1];
  }

  std::cout << std::setprecision(12)
            << "Problem: block-sparse bundle adjustment (20 cameras, 200 points)\n"
            << "Runs: " << runs << '\n'
            << "Initial cost: " << initial_cost << '\n'
            << "Final cost: " << final_cost << '\n'
            << "Iterations: " << iterations << '\n'
            << "Camera 1: " << final_camera.transpose() << '\n'
            << "Point 1: " << final_point.transpose() << '\n';
  return 0;
}