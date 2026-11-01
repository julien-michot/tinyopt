// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <vector>

#include <tinyopt/tinyopt.h>

/// Custom 2D landmark parameter type used by the tutorial compile checks.
template <typename T>
struct Landmark2D {
  using Scalar = T;
  static constexpr tinyopt::Index Dims = 2;

  tinyopt::Vector<T, 2> position = tinyopt::Vector<T, 2>::Zero();

  Landmark2D &operator+=(const tinyopt::Vector<T, Dims> &delta) {
    position += delta;
    return *this;
  }

  template <typename T2>
  Landmark2D<T2> cast() const {
    return {position.template cast<T2>()};
  }
};

/// Custom 2D pose parameter type used by the tutorial compile checks.
template <typename T>
struct Pose2 {
  using Scalar = T;
  using Vec2 = tinyopt::Vector<T, 2>;
  using Tangent = tinyopt::Vector<T, 3>;
  static constexpr tinyopt::Index Dims = 3;

  Pose2() : translation(Vec2::Zero()), orientation(Vec2(1, 0)) {}

  template <typename T2>
  static auto cast(const Pose2 &pose) {
    Pose2<T2> result;
    result.translation = pose.translation.template cast<T2>();
    result.orientation = pose.orientation.template cast<T2>();
    return result;
  }

  Pose2 &operator+=(const Tangent &delta) {
    translation += delta.template head<2>();

    const T tangent = delta[2];
    const T denominator = T(1) + tangent * tangent;
    const T cosine = (T(1) - tangent * tangent) / denominator;
    const T sine = T(2) * tangent / denominator;
    const Vec2 previous_orientation = orientation;
    orientation[0] = cosine * previous_orientation[0] - sine * previous_orientation[1];
    orientation[1] = sine * previous_orientation[0] + cosine * previous_orientation[1];
    return *this;
  }

  Vec2 translation;
  Vec2 orientation;
};

namespace tinyopt::traits {

template <typename T>
struct params_trait<Pose2<T>> {
  using Scalar = T;
  static constexpr Index Dims = 3;

  template <typename T2>
  static Pose2<T2> cast(const Pose2<T> &pose) {
    Pose2<T2> result;
    result.translation = pose.translation.template cast<T2>();
    result.orientation = pose.orientation.template cast<T2>();
    return result;
  }

  static void PlusEq(Pose2<T> &pose, const Vector<T, Dims> &delta) {
    pose.translation += delta.template head<2>();
    const T tangent = delta[2];
    const T denominator = T(1) + tangent * tangent;
    const T cosine = (T(1) - tangent * tangent) / denominator;
    const T sine = T(2) * tangent / denominator;
    const auto previous_orientation = pose.orientation;
    pose.orientation[0] = cosine * previous_orientation[0] - sine * previous_orientation[1];
    pose.orientation[1] = sine * previous_orientation[0] + cosine * previous_orientation[1];
  }
};

}  // namespace tinyopt::traits

/// Parameter type with a fixed component used by the tutorial compile checks.
template <typename T = double>
struct TutorialFixedParams {
  using Scalar = T;
  static constexpr tinyopt::Index Dims = 3;
  tinyopt::Vector<T, 3> values{T(0.0), T(5.0), T(0.0)};
  std::vector<tinyopt::Index> fixed() const { return {1}; }  // Index 1 is fixed at 5.0
  template <typename DeltaType>
  TutorialFixedParams &operator+=(const DeltaType &delta) {
    values += delta;
    return *this;
  }
  template <typename T2>
  TutorialFixedParams<T2> cast() const {
    TutorialFixedParams<T2> p;
    p.values = values.template cast<T2>();
    return p;
  }
};

/// Parameter type with reduced head parameters used by the Schur tutorial example.
struct TutorialSchurParams {
  using Scalar = double;
  static constexpr tinyopt::Index Dims = 4;
  tinyopt::Vector<double, 4> values{0.0, 0.0, 0.0, 0.0};
  std::pair<tinyopt::Index, tinyopt::Index> reduced() const {
    return {0, 2};  // 2 reduced head parameters
  }
  template <typename DeltaType>
  TutorialSchurParams &operator+=(const DeltaType &delta) {
    values += delta;
    return *this;
  }
  template <typename T2>
  TutorialSchurParams cast() const {
    return *this;
  }
};
