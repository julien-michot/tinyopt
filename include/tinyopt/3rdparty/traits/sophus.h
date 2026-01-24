// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <sophus/se3.hpp>  // for now we define only SE3 and SO3

#include <tinyopt/traits.h>

namespace tinyopt::traits {

// Sophus's SE3<T>
template <typename T>
struct params_trait<Sophus::SE3<T>> {
  using Scalar = T;                                 // The scalar type
  static constexpr Index Dims = Sophus::SE3<T>::DoF;  // Compile-time parameters dimensions

  template <typename T2>
  static Sophus::SE3<T2> cast(const Sophus::SE3<T> &pose) {
    return pose.template cast<T2>();
  }

  // Define update / manifold
  static void PlusEq(Sophus::SE3<T> &pose, const Eigen::Vector<Scalar, Dims> &delta) {
    pose *= Sophus::SE3<T>::exp(delta);  // right update
  }
};

// Sophus's SO3<T>
template <typename T>
struct params_trait<Sophus::SO3<T>> {
  using Scalar = T;                                 // The scalar type
  static constexpr Index Dims = Sophus::SO3<T>::DoF;  // Compile-time parameters dimensions

  template <typename T2>
  static Sophus::SO3<T2> cast(const Sophus::SO3<T> &rot) {
    return rot.template cast<T2>();
  }

  // Define update / manifold
  static void PlusEq(Sophus::SO3<T> &rot, const Eigen::Vector<Scalar, Dims> &delta) {
    rot *= Sophus::SO3<T>::exp(delta);  // right update
  }
};

}  // namespace tinyopt::traits