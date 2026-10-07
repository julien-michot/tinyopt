// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/sparse_optimizer_terminate_action.h>

namespace tinyopt::benchmark {

class G2oTerminationAction final : public g2o::SparseOptimizerTerminateAction {
 public:
  explicit G2oTerminationAction(int max_iterations) {
    setGainThreshold(1e-6);
    setMaxIterations(max_iterations);
  }

  bool Converged() const { return converged_; }

  g2o::HyperGraphAction* operator()(const g2o::HyperGraph* graph,
                                    Parameters* parameters = nullptr) override {
    const auto* iteration =
        dynamic_cast<const ParametersIteration*>(parameters);
    const double previous_cost = _lastChi;
    g2o::HyperGraphAction* result =
        g2o::SparseOptimizerTerminateAction::operator()(graph, parameters);
    if (iteration != nullptr && iteration->iteration > 0) {
      const auto* optimizer = static_cast<const g2o::SparseOptimizer*>(graph);
      const double current_cost = optimizer->activeRobustChi2();
      const double denominator =
          std::max(std::abs(previous_cost), std::numeric_limits<double>::min());
      const double relative_decrease = (previous_cost - current_cost) / denominator;
      const std::vector<double> current_estimates = Snapshot(*optimizer);
      double step_norm_squared = 0;
      if (current_estimates.size() == previous_estimates_.size()) {
        for (std::size_t index = 0; index < current_estimates.size(); ++index) {
          const double step = current_estimates[index] - previous_estimates_[index];
          step_norm_squared += step * step;
        }
      }
      converged_ = (relative_decrease >= 0 && relative_decrease < 1e-6) ||
                   step_norm_squared < 1e-16;
      if (converged_) setOptimizerStopFlag(optimizer, true);
      previous_estimates_ = current_estimates;
    } else if (iteration != nullptr && iteration->iteration == 0) {
      previous_estimates_ = Snapshot(*static_cast<const g2o::SparseOptimizer*>(graph));
    }
    return result;
  }

 private:
  static std::vector<double> Snapshot(const g2o::SparseOptimizer& optimizer) {
    std::vector<double> estimates;
    for (const auto* vertex : optimizer.activeVertices()) {
      std::vector<double> vertex_estimate;
      if (!vertex->getMinimalEstimateData(vertex_estimate))
        throw std::runtime_error("g2o vertex does not expose its minimal estimate");
      estimates.insert(estimates.end(), vertex_estimate.begin(), vertex_estimate.end());
    }
    return estimates;
  }

  bool converged_ = false;
  std::vector<double> previous_estimates_;
};

}  // namespace tinyopt::benchmark
