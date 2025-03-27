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

#include <sophus/se3.hpp>  // for now we define only SE3 and SO3

#include <tinyopt/traits.h>

namespace tinyopt::traits {

// Sophus's SE3<T>
template <typename T>
struct params_trait<Sophus::SE3<T>> {
  using Scalar = T;                                 // The scalar type
  static constexpr int Dims = Sophus::SE3<T>::DoF;  // Compile-time parameters dimensions

  template <typename T2>
  static Sophus::SE3<T2> cast(const Sophus::SE3<T> &pose) {
    return pose.template cast<T2>();
  }

  // Define update / manifold
  static void pluseq(Sophus::SE3<T> &pose, const Eigen::Vector<Scalar, Dims> &delta) {
    pose *= Sophus::SE3<T>::exp(delta);  // right update
  }
};

// Sophus's SO3<T>
template <typename T>
struct params_trait<Sophus::SO3<T>> {
  using Scalar = T;                                 // The scalar type
  static constexpr int Dims = Sophus::SO3<T>::DoF;  // Compile-time parameters dimensions

  template <typename T2>
  static Sophus::SO3<T2> cast(const Sophus::SO3<T> &rot) {
    return rot.template cast<T2>();
  }

  // Define update / manifold
  static void pluseq(Sophus::SO3<T> &rot, const Eigen::Vector<Scalar, Dims> &delta) {
    rot *= Sophus::SO3<T>::exp(delta);  // right update
  }
};

}  // namespace tinyopt::traits