// Copyright (C) 2025 Julien Michot. All Rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <tinyopt/math.h>

namespace tinyopt {

/***
 *  @brief Struct containing a cost and related informations
 *
 ***/
struct Cost {
  using Scalar = double;
  Cost(Scalar cost_ = Scalar(0), int num_resisuals_ = 1, float inlier_ratio_ = 1.0f,
       const std::string &log_ = "")
      : cost{cost_}, num_resisuals{num_resisuals_}, inlier_ratio{inlier_ratio_}, log_str{log_} {}

  // Constructor receiving a 'residuals' Vector/Matrix. The cost will be the L2/Frobenius norm.
  template <typename Derived>
  Cost(const MatrixBase<Derived> &residuals, float inlier_ratio_ = 1.0f,
       const std::string &log_ = "")
      : Cost(residuals.squaredNorm(), (int)residuals.size(), inlier_ratio_, log_) {}

  operator bool() const { return isValid(); }
  operator Scalar() const { return cost; }

  // Comparisons
  bool operator<(const Cost &other) const { return cost < other.cost; }
  bool operator<=(const Cost &other) const { return cost <= other.cost; }
  bool operator<(Scalar other_cost) const { return cost < other_cost; }
  bool operator<=(Scalar other_cost) const { return cost <= other_cost; }

  /// Accumulate another cost
  Cost &operator+=(const Cost &other) {
    cost += other.cost;
    if (num_resisuals + other.num_resisuals > 0)
      inlier_ratio =
          (NumInliers() + other.NumInliers()) / (float)(num_resisuals + other.num_resisuals);
    num_resisuals += other.num_resisuals;
    if (!other.log_str.empty()) log_str += " " + other.log_str;
    return *this;
  }

  friend std::ostream &operator<<(std::ostream &os, const Cost &cost) {
    os << "ε:" << cost.cost << ", n:" << cost.num_resisuals << ", in:" << cost.inlier_ratio * 100.0f
       << "%";
    if (!cost.log_str.empty()) os << cost.log_str;
    return os;
  }

  bool isValid() const { return num_resisuals > 0 && cost != std::numeric_limits<Scalar>::max(); }
  int NumInliers() const { return (int)(num_resisuals * inlier_ratio); }
  int NumOutliers() const { return (int)(num_resisuals * (1.0f - inlier_ratio)); }

  Scalar cost;         ///< The function cost
  int num_resisuals;   ///< The number of residuals (for e.g. NLLS)
  float inlier_ratio;  ///< The ratio of inlier residuals (when robust norms are used)
  std::string
      log_str;  ///< Extra information that will be printed as part of the optimization iterations
};

}  // namespace tinyopt
