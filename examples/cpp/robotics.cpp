// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cmath>
#include <iostream>

// Estimate a robot sensor's 2D translation by aligning known local landmarks
// with corresponding map points. The sensor yaw is assumed to be known.
#include <tinyopt/optimizers/gd.h>

// Tinyopt provides fixed-size vectors, Options, and the gradient-descent optimizer.
using namespace tinyopt;

int main() {
  // Local-frame landmark coordinates observed by the sensor.
  const std::array<Vec2, 4> landmarks{Vec2(-1.0, -0.5), Vec2(0.8, -0.3), Vec2(-0.4, 0.9),
                                      Vec2(1.2, 0.7)};
  // This example isolates translation estimation by taking the sensor yaw as known.
  constexpr double sensor_yaw = 0.35;
  const double cosine = std::cos(sensor_yaw);
  const double sine = std::sin(sensor_yaw);
  const Vec2 true_translation(1.2, -0.7);
  // Transform the landmarks into map coordinates to create deterministic observations.
  std::array<Vec2, 4> map_points;
  for (std::size_t i = 0; i < landmarks.size(); ++i) {
    const Vec2 rotated(cosine * landmarks[i].x() - sine * landmarks[i].y(),
                       sine * landmarks[i].x() + cosine * landmarks[i].y());
    map_points[i] = true_translation + rotated;
  }

  // Translation is the only unknown; start from the origin.
  Vec2 translation = Vec2::Zero();
  // Sum squared point-alignment errors; Tinyopt autodifferentiates this scalar objective.
  const auto objective = [&](const auto& estimate) {
    auto error = estimate[0] * 0.0;
    for (std::size_t i = 0; i < landmarks.size(); ++i) {
      const Vec2 rotated(cosine * landmarks[i].x() - sine * landmarks[i].y(),
                         sine * landmarks[i].x() + cosine * landmarks[i].y());
      const auto residual = estimate + rotated - map_points[i];
      error += residual.squaredNorm();
    }
    return error;
  };

  // Gradient descent is a first-order choice for this smooth least-squares objective.
  Options options;
  options.log.enable = false;
  options.stop.max_iters = 300;
  options.stop.min_rerr_dec = 0;
  options.gd.lr = 0.05f;
  // Vec2 selects a two-coordinate gradient workspace; Optimize updates translation in place.
  const auto summary = gd::Optimizer<Vec2>(options)(translation, objective);
  std::cout << "Estimated sensor translation: " << translation.transpose() << '\n'
            << "Optimization succeeded: " << std::boolalpha << summary.Succeeded() << '\n';
  // Return a nonzero exit code if the solver reports failure.
  return summary.Succeeded() ? 0 : 1;
}