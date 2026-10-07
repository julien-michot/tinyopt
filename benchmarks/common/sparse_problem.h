// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tinyopt::benchmark::sparse_problem {

inline double Target(int index) { return 0.1 * static_cast<double>(index % 17) - 0.8; }

inline double Initial(int index) { return 1.5 - 0.03 * static_cast<double>(index % 19); }

inline double DifferenceTarget(int index) { return Target(index) - Target(index - 1); }

}  // namespace tinyopt::benchmark::sparse_problem
