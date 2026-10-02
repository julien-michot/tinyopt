// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <tuple>
#include <type_traits>

#if CATCH2_VERSION == 2
#include <catch2/catch.hpp>
#else
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#endif

#include <tinyopt/diff/jet.h>
#include <tinyopt/traits/params_pack.h>

using Catch::Approx;
using namespace tinyopt;

TEST_CASE("tinyopt_parameter_pack_keeps_fixed_parameters_by_reference") {
  double scale = 2.0;
  Vec3 position(3.0, 4.0, 5.0);
  using Pack = traits::detail::ParamsPack<double &, Vec3 &>;
  Pack pack(scale, position);

  static_assert(traits::params_trait<Pack>::Dims == 4);
  static_assert(std::is_same_v<std::tuple_element_t<0, decltype(pack.values)>, double &>);
  static_assert(std::is_same_v<std::tuple_element_t<1, decltype(pack.values)>, Vec3 &>);
  REQUIRE(traits::params_trait<Pack>::dims(pack) == 4);

  traits::params_trait<Pack>::PlusEq(pack, Vec4(1.0, 2.0, 3.0, 4.0));
  REQUIRE(scale == Approx(3.0));
  REQUIRE((position - Vec3(5.0, 7.0, 9.0)).norm() == Approx(0.0).margin(1e-12));

  using Jet = diff::Jet<double, 4>;
  const auto casted = traits::params_trait<Pack>::template cast<Jet>(pack);
  static_assert(traits::params_trait<std::decay_t<decltype(casted)>>::Dims == 4);
  REQUIRE(std::get<0>(casted.values).a == Approx(scale));
  REQUIRE(std::get<1>(casted.values)(0).a == Approx(position(0)));
}