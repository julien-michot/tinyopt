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

namespace lieplusplus { // placing it in a namespace
#include <groups/SEn3.hpp>  // for now we define only SE3 and SO3
}

#include <tinyopt/traits.h>

namespace tinyopt::traits {

template <typename T, int n>
struct params_trait<lieplusplus::group::SEn3<T, n>> {
  using Pose = lieplusplus::group::SEn3<T, n>;
  using Scalar = T;  // The scalar type
  static constexpr int Dims =
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
  static constexpr int Dims =
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