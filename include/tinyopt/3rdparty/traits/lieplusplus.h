// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <Eigen/Dense> // Must be before any include groups/* due to lieplusplus namespace

namespace lieplusplus { // Place all lie++ under the lieplusplus namespace
#include <groups/SEn3.hpp>  // For now we define only SE3 and SO3
}

#include <tinyopt/traits.h>

namespace tinyopt::traits {

template <typename T, int n>
struct params_trait<lieplusplus::group::SEn3<T, n>> {
  using Pose = lieplusplus::group::SEn3<T, n>;
  using Scalar = T;  // The scalar type
  static constexpr Index Dims =
      Pose::VectorType::RowsAtCompileTime;  // Compile-time parameters dimensions

  template <typename T2>
  static lieplusplus::group::SEn3<T2, n> cast(const Pose &pose) {
    std::array<Eigen::Matrix<T2, 3, 1>, n> iso;
    for (int i = 0; i < n; ++i)
      iso[i] = pose.t()[i].template cast<T2>();
    return lieplusplus::group::SEn3<T2, n>(pose.q().template cast<T2>(), iso);
  }

  // Define update / manifold
  static void PlusEq(Pose &pose, const Eigen::Vector<Scalar, Dims> &delta) {
    pose = pose * Pose::exp(delta);  // right update
  }
};

template <typename T>
struct params_trait<lieplusplus::group::SO3<T>> {
  using Pose = lieplusplus::group::SO3<T>;
  using Scalar = T;  // The scalar type
  static constexpr Index Dims =
      Pose::VectorType::RowsAtCompileTime;  // Compile-time parameters dimensions

  template <typename T2>
  static lieplusplus::group::SO3<T2> cast(const Pose &pose) {
    return lieplusplus::group::SO3<T2>(pose.q().template cast<T2>());
  }

  // Define update / manifold
  static void PlusEq(Pose &pose, const Eigen::Vector<Scalar, Dims> &delta) {
    pose = pose * Pose::exp(delta);  // right update
  }
};

}  // namespace tinyopt::traits