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

#include <tinyopt/math.h>
#include <tinyopt/traits.h>
#include <type_traits>

namespace tinyopt {

template <typename P0, typename P1>
struct ParamsPack2 : std::tuple<P0, P1> {
  using ParamTrait0 = traits::params_trait<std::decay_t<P0>>;
  using ParamTrait1 = traits::params_trait<std::decay_t<P1>>;
  static constexpr Index Dims0 = ParamTrait0::Dims;
  static constexpr Index Dims1 = ParamTrait1::Dims;
  static constexpr Index Dims = (Dims0 == Dynamic || Dims1 == Dynamic) ? Dynamic : Dims0 + Dims1;

  using Scalar0 = typename ParamTrait0::Scalar;
  using Scalar1 = typename ParamTrait1::Scalar;
  using Scalar = std::conditional_t<sizeof(Scalar0) >= sizeof(Scalar1), Scalar0, Scalar1>;

  // Constructors
  template <typename P = P0, std::enable_if_t<std::is_reference_v<P>, int> = 0>
  ParamsPack2(const std::tuple<P0, P1> &_ps) : ps{_ps} {}
  // Move constructor
  template <typename P = P0, std::enable_if_t<!std::is_reference_v<P>, int> = 0>
  ParamsPack2(std::tuple<P0, P1> &&_ps) : ps{std::move(_ps)} {}

  template <typename P = P0, std::enable_if_t<std::is_reference_v<P>, int> = 0>
  ParamsPack2(P0 _p0, P1 _p1) : ps{std::make_tuple<P0, P1>(_p0, _p1)} {}

  // Constructor with a reference if !is_reference_v<P>
  template <typename P = P0, std::enable_if_t<!std::is_reference_v<P>, int> = 0>
  ParamsPack2(P0 &_p0, P1 &_p1) : ps{std::make_tuple<P0, P1>(_p0, _p1)} {}

  // Move constructor
  template <typename P = P0, std::enable_if_t<std::is_reference_v<P>, int> = 0>
  ParamsPack2(P0 &_p0, P1 &_p1) : ps{std::make_tuple<P0, P1>(std::move(_p0), std::move(_p1))} {}

  int dims(int i) const { return i == 0 ? ParamTrait0::dims(p0()) : ParamTrait1::dims(p1()); }
  int dims() const { return dims(0) + dims(1); }

  // Returns a copy where the scalar is converted to another type 'T2'.
  // This is only used by auto differentiation
  template <typename T2>
  inline auto cast() const {
    auto ps_ =
        std::make_tuple(ParamTrait0::template cast<T2>(p0()), ParamTrait1::template cast<T2>(p1()));
    using P0_ = std::tuple_element_t<0, decltype(ps_)>;
    using P1_ = std::tuple_element_t<1, decltype(ps_)>;
    return ParamsPack2<P0_, P1_>(std::move(ps_));
  }
  // Update / manifold
  ParamsPack2 &operator+=(const auto &delta) {
    if constexpr (Dims0 == Dynamic)
      ParamTrait0::PlusEq(p0(), delta.head(dims(0)).template cast<Scalar0>().eval());
    else
      ParamTrait0::PlusEq(p0(), delta.template head<Dims0>().template cast<Scalar0>().eval());
    if constexpr (Dims1 == Dynamic)
      ParamTrait1::PlusEq(p1(), delta.tail(dims(1)).template cast<Scalar1>().eval());
    else
      ParamTrait1::PlusEq(p1(), delta.template tail<Dims1>().template cast<Scalar1>().eval());
    return *this;
  }

  auto p0() const { return std::get<0>(ps); }
  auto p0() { return std::get<0>(ps); }
  auto p1() const { return std::get<1>(ps); }
  auto p1() { return std::get<1>(ps); }

  std::tuple<P0, P1> ps;
};

}  // namespace tinyopt
