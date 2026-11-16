// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <limits>
#include <type_traits>
#include <utility>

#include <tinyopt/tinyopt.h>

namespace tinyopt_bounds_test {

using namespace tinyopt;

template <typename T>
struct AnglePose {
  using Scalar = T;
  static constexpr Index Dims = 1;
  T angle = T(0);

  auto bounds() const { return std::make_pair(T(-0.5) - angle, T(0.5) - angle); }

  AnglePose& operator+=(const Vector<T, 1>& delta) {
    angle += delta[0];
    return *this;
  }

  template <typename T2>
  AnglePose<T2> cast() const {
    return {T2(angle)};
  }
};

template <typename T = double>
struct DynamicBoundedState {
  using Scalar = T;
  static constexpr Index Dims = Dynamic;
  Vector<T, Dynamic> values;

  explicit DynamicBoundedState(Index dims) : values(Vector<T, Dynamic>::Zero(dims)) {}

  Index dims() const { return values.size(); }

  auto bounds() const {
    const T inf = std::numeric_limits<T>::infinity();
    Vector<T, Dynamic> lower = Vector<T, Dynamic>::Constant(values.size(), -inf);
    Vector<T, Dynamic> upper = Vector<T, Dynamic>::Constant(values.size(), inf);
    upper[0] = T(0.5);
    return std::pair{lower, upper};
  }

  DynamicBoundedState& operator+=(const Vector<T, Dynamic>& delta) {
    values += delta;
    return *this;
  }

  template <typename T2>
  DynamicBoundedState<T2> cast() const {
    DynamicBoundedState<T2> result(values.size());
    result.values = values.template cast<T2>();
    return result;
  }
};

template <typename T>
struct BoundedScalar {
  using Scalar = T;
  static constexpr Index Dims = 1;
  T value = T(0);

  auto bounds() const { return std::pair{-T(0.25), T(0.25)}; }

  BoundedScalar& operator+=(const Vector<T, 1>& delta) {
    value += delta[0];
    return *this;
  }

  template <typename T2>
  BoundedScalar<T2> cast() const {
    return {static_cast<T2>(value)};
  }
};

}  // namespace tinyopt_bounds_test
