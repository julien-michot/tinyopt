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

#include <ceres/jet.h>

namespace tinyopt {

template <typename T, int N> using Jet = ceres::Jet<T, N>;

/// Add convenient operators with scalar

template <typename T, int N> Jet<T, N> operator*(const Jet<T, N> &jet, T v) {
  return jet * Jet<T, N>(v);
}
template <typename T, int N> Jet<T, N> operator/(const Jet<T, N> &jet, T v) {
  return jet / Jet<T, N>(v);
}
template <typename T, int N> Jet<T, N> operator+(const Jet<T, N> &jet, T v) {
  return jet + Jet<T, N>(v);
}
template <typename T, int N> Jet<T, N> operator-(const Jet<T, N> &jet, T v) {
  return jet - Jet<T, N>(v);
}

template <typename T, int N> Jet<T, N> operator*(T v, const Jet<T, N> &jet) {
  return Jet<T, N>(v) * jet;
}
template <typename T, int N> Jet<T, N> operator/(T v, const Jet<T, N> &jet) {
  return Jet<T, N>(v) / jet;
}
template <typename T, int N> Jet<T, N> operator+(T v, const Jet<T, N> &jet) {
  return Jet<T, N>(v) + jet;
}
template <typename T, int N> Jet<T, N> operator-(T v, const Jet<T, N> &jet) {
  return Jet<T, N>(v) - jet;
}

} // namespace tinyopt
