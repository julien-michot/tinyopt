// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <iostream>
#include <string>

namespace tinyopt::benchmark {

inline void PrintIterations(const std::string& category, const std::string& problem,
                            const std::string& backend, int iterations, bool converged) {
  std::cout << "[iterations] " << category << " | " << problem << " | " << backend << " | "
            << iterations << " | " << (converged ? "ok" : "no") << '\n';
}

}  // namespace tinyopt::benchmark
