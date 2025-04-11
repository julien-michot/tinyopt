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

#include <chrono>

namespace tinyopt {

using Clock = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::high_resolution_clock::time_point;

/**
 * @brief Returns the current time point using the high-resolution clock.
 *
 * This function provides a convenient way to capture the current time for
 * benchmarking or timing purposes.
 *
 * @return The current time point as a `std::chrono::time_point<Clock>`.
 */
inline auto tic() { return Clock::now(); }

/**
 * @brief Calculates the elapsed time in milliseconds between a given time point and the current
 * time.
 *
 * This function measures the duration between a previously recorded time point (`t0`)
 * and the current time, returning the result in milliseconds. It utilizes the
 * high-resolution clock for precise timing.
 *
 * @param t0 The starting time point (`std::chrono::time_point<Clock>`).
 * @return The elapsed time in milliseconds (double).
 */
inline double toc_ms(const std::chrono::time_point<Clock> &t0) {
  return static_cast<double>(
             std::chrono::duration_cast<std::chrono::microseconds>(tic() - t0).count()) *
         1e-3;
}

/**
 * @brief Calculates the elapsed time in milliseconds between two given time points.
 *
 * This function measures the duration between two specified time points (`t0` and `t1`),
 * returning the result in milliseconds. It relies on the high-resolution clock for
 * accurate timing.
 *
 * @param t0 The starting time point (`std::chrono::time_point<Clock>`).
 * @param t1 The ending time point (`std::chrono::time_point<Clock>`).
 * @return The elapsed time in milliseconds (double).
 */
inline double dt_ms(const std::chrono::time_point<Clock> &t0,
                    const std::chrono::time_point<Clock> &t1) {
  return static_cast<double>(
             std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()) *
         1e-3;
}

}  // namespace tinyopt
