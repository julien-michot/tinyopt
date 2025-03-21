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

/// Basically Ceres'Jet with support for *=, +=, ... scalar
template <typename T, int N>
struct Jet : ceres::Jet<T, N> {
  using Base = ceres::Jet<T, N>;

  Jet() : Base() {}
  explicit Jet(const Base& b) :Base(b) {}
  explicit Jet(const T& v) :Base(v) {}

  // TODO avoid this copy
  Jet &operator=(const Base &jet) {
    this->a = jet.a;
    this->v = jet.v;
    return *this;
  }

  Jet operator*(const Jet &jet) const {
    Jet out = *this;
    out *= jet;
    return out;
  }
  Jet operator+(const Jet &jet) const {
    Jet out = *this;
    out += jet;
    return out;
  }
  Jet operator-(const Jet &jet) const {
    Jet out = *this;
    out -= jet;
    return out;
  }

  Jet &operator*=(T v) {
    Base::operator*=(Base(v));
    return *this;
  }
  Jet &operator/=(T v) {
    Base::operator/=(Base(v));
    return *this;
  }
  Jet &operator+=(T v) {
    Base::operator+=(Base(v));
    return *this;
  }
  Jet &operator-=(T v) {
    Base::operator-=(Base(v));
    return *this;
  }

  Jet &operator*=(const Jet &jet) {
    Base::operator*=(jet.base());
    return *this;
  }
  Jet &operator/=(const Jet &jet) {
    Base::operator/=(jet.base());
    return *this;
  }
  Jet &operator+=(const Jet &jet) {
    Base::operator+=(jet.base());
    return *this;
  }
  Jet &operator-=(const Jet &jet) {
    Base::operator-=(jet.base());
    return *this;
  }
  const Base &base() const { return *this; }
};

template <typename T, int N>
Jet<T, N> operator*(T v, const Jet<T, N> &jet) {
  return Jet<T, N>(v) * jet;
}
template <typename T, int N>
Jet<T, N> operator/(T v, const Jet<T, N> &jet) {
  return Jet<T, N>(v) / jet;
}
template <typename T, int N>
Jet<T, N> operator+(T v, const Jet<T, N> &jet) {
  return Jet<T, N>(v) + jet;
}
template <typename T, int N>
Jet<T, N> operator-(T v, const Jet<T, N> &jet) {
  return Jet<T, N>(v) - jet;
}

} // namespace tinyopt
