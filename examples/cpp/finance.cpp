// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <type_traits>

// This example chooses a long-only portfolio with a target expected return.
// L-BFGS minimizes portfolio variance while a softmax parameterization keeps
// all asset weights nonnegative and summing to one.
#include <tinyopt/optimizers/bfgs.h>

// Tinyopt provides the fixed-size vectors/matrices and optimizer APIs below.
using namespace tinyopt;

int main() {
  // Two free logits determine three asset weights; the third score is fixed at 1.
  Vec2 allocation_logits = Vec2::Zero();
  // Covariance controls portfolio risk; expected_returns defines the return model.
  Mat3 covariance;
  covariance << 0.040, 0.006, 0.004, 0.006, 0.090, 0.010, 0.004, 0.010, 0.160;
  const Vec3 expected_returns(0.05, 0.09, 0.14);
  constexpr double target_return = 0.09;

  // Tinyopt differentiates this scalar objective automatically with respect to logits.
  const auto objective = [&](const auto& logits) {
    using std::exp;
    Eigen::Matrix<typename std::decay_t<decltype(logits)>::Scalar, 3, 1> scores;
    scores[0] = exp(logits[0]);
    scores[1] = exp(logits[1]);
    scores[2] = typename std::decay_t<decltype(logits)>::Scalar(1.0);
    const auto allocation = scores / scores.sum();
    const auto budget_error = allocation.sum() - 1.0;
    const auto return_error = expected_returns.dot(allocation) - target_return;
    return 0.5 * allocation.dot(covariance * allocation) + 100.0 * budget_error * budget_error +
           100.0 * return_error * return_error;
  };

  // Options configure this solve without changing the library-wide defaults.
  Options options;
  options.log.enable = false;          // Keep the example output focused on its result.
  options.stop.max_iters = 2000;       // Allow the line-search method enough iterations.
  options.stop.min_rerr_dec = 0;       // Do not stop early on a small relative decrease.
  options.lbfgs.step_size = 0.1f;       // Initial step size for L-BFGS.
  options.lbfgs.max_step_size = 0.5f;   // Bound line-search growth.

  // The optimizer updates allocation_logits in place and returns a solve summary.
  const auto summary = lbfgs::Optimizer<Vec2>(options)(allocation_logits, objective);
  // Convert the fitted logits back to portfolio weights for reporting.
  Vec3 scores(std::exp(allocation_logits[0]), std::exp(allocation_logits[1]), 1.0);
  const Vec3 weights = scores / scores.sum();
  std::cout << "Long-only portfolio weights: " << weights.transpose() << "\n"
            << "Expected return: " << expected_returns.dot(weights) << "\n"
            << "Optimization succeeded: " << std::boolalpha << summary.Succeeded() << '\n';
  // Return a failing process status if Tinyopt could not complete the solve.
  return summary.Succeeded() ? 0 : 1;
}