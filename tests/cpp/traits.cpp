// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <type_traits>

#if CATCH2_VERSION == 2
#include <catch2/catch.hpp>
#else
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#endif

#include <tinyopt/tinyopt.h>
#include <tinyopt/traits.h>

using namespace tinyopt;
using namespace tinyopt::nlls;

TEST_CASE("tinyopt_params_traits_stl") {
  SECTION("std::array<float, 5>") {
    using Params = std::array<float, 5>;
    Params x;
    REQUIRE(traits::params_trait<Params>::Dims == 5);
    REQUIRE(traits::params_trait<Params>::dims(x) == 5);
  }
  SECTION("std::vector<float>") {
    using Params = std::vector<double>;
    Params x{{0.4, 0.5}};
    REQUIRE(traits::params_trait<Params>::Dims == Dynamic);
    REQUIRE(traits::params_trait<Params>::dims(x) == 2);
  }
  SECTION("std::array<Vec4, 5>") {
    using Params = std::array<Vec4, 5>;
    Params x;
    REQUIRE(traits::params_trait<Params>::Dims == 4 * 5);
    REQUIRE(traits::params_trait<Params>::dims(x) == 4 * 5);
  }
  SECTION("std::array<VecX, 2>") {
    using Params = std::array<VecX, 2>;
    using ptraits = traits::params_trait<Params>;
    Params x;
    REQUIRE(ptraits::Dims == Dynamic);
    REQUIRE(ptraits::dims(x) == 0);
    static_assert(std::is_same_v<std::decay_t<decltype(ptraits::template cast<float>(x))>,
                                 std::array<VecXf, 2>>,
                  "Wrong casting");
  }
  SECTION("std::vector<Vec2>") {
    using Params = std::vector<Vec2>;
    using ptraits = traits::params_trait<Params>;
    Params x{{Vec2::Zero(), Vec2::Zero(), Vec2::Zero()}};
    REQUIRE(ptraits::Dims == Dynamic);
    REQUIRE(ptraits::dims(x) == 6);
    static_assert(std::is_same_v<std::decay_t<decltype(ptraits::template cast<float>(x))>,
                                 std::vector<Vec2f>>,
                  "Wrong casting");
  }
  SECTION("std::pair<Vec2, Vec3>") {
    using Params = std::pair<Vec2, Vec3>;
    using ptraits = traits::params_trait<Params>;
    Params x;
    REQUIRE(ptraits::Dims == 2 + 3);
    REQUIRE(ptraits::dims(x) == 2 + 3);
    static_assert(std::is_same_v<std::decay_t<decltype(ptraits::template cast<float>(x))>,
                                 std::pair<Vec2f, Vec3f>>,
                  "Wrong casting");
  }
  SECTION("std::pair<Vec2, VecX>") {
    using Params = std::pair<Vec2, VecX>;
    Params x = std::make_pair(Vec2::Zero(), VecX::Random(4));
    REQUIRE(traits::params_trait<Params>::Dims == Dynamic);
    REQUIRE(traits::params_trait<Params>::dims(x) == 2 + 4);
  }
  SECTION("std::pair<vector<float>, Vec3>") {
    using Params = std::pair<std::vector<float>, Vec3>;
    Params x = std::make_pair(std::vector<float>{{1, 2, 3, 4}}, Vec3::Zero());
    REQUIRE(traits::params_trait<Params>::Dims == Dynamic);
    REQUIRE(traits::params_trait<Params>::dims(x) == 4 + 3);
  }
  SECTION("std::pair<std::vector<Vec3>, std::array<VecX, 4>>") {
    using Params = std::pair<std::vector<Vec3>, std::array<VecX, 4>>;
    std::vector<Vec3> a{Vec3::Zero(), Vec3::Zero()};
    std::array<VecX, 4> b{{VecX::Random(5), VecX::Random(2), VecX::Random(0), VecX::Random(0)}};
    Params x = std::make_pair(a, b);
    REQUIRE(traits::params_trait<Params>::Dims == Dynamic);
    REQUIRE(traits::params_trait<Params>::dims(x) == 6 + 7);
  }
  SECTION("std::pair<std::array<Vec3>, std::array<Vec2, 4>>") {
    using Params = std::pair<std::array<Vec3, 5>, std::array<Vec2, 4>>;
    std::array<Vec3, 5> a;
    std::array<Vec2, 4> b;
    Params x = std::make_pair(a, b);
    REQUIRE(traits::params_trait<Params>::Dims == 15 + 8);
    REQUIRE(traits::params_trait<Params>::dims(x) == 15 + 8);
  }
}

namespace {
struct CustomWithNorm2 {
  double norm2() const { return 42.0; }
};

struct CustomWithoutNorm2 {};
}  // namespace

TEST_CASE("tinyopt_params_traits_norm2") {
  SECTION("scalar") {
    static_assert(has_norm2_v<float>);
    static_assert(has_norm2_v<double>);
    static_assert(has_norm2_v<int>);

    const float f = 3.0f;
    REQUIRE(traits::norm2(f) == Catch::Approx(9.0f));
    const double d = -4.0;
    REQUIRE(traits::norm2(d) == Catch::Approx(16.0));
    REQUIRE(tinyopt::norm2(d) == Catch::Approx(16.0));
  }

  SECTION("Eigen::Matrix and Vector") {
    static_assert(has_norm2_v<Vec3>);
    static_assert(has_norm2_v<Mat2>);
    static_assert(has_norm2_v<VecX>);

    const Vec3 v(1.0, 2.0, 3.0);
    REQUIRE(traits::norm2(v) == Catch::Approx(14.0));
    REQUIRE(tinyopt::norm2(v) == Catch::Approx(14.0));

    Mat2 m;
    m << 1.0, 2.0, 3.0, 4.0;
    REQUIRE(traits::norm2(m) == Catch::Approx(30.0));

    VecX vx(3);
    vx << 2.0, -1.0, 3.0;
    REQUIRE(traits::norm2(vx) == Catch::Approx(14.0));
  }

  SECTION("std::vector") {
    static_assert(has_norm2_v<std::vector<double>>);
    static_assert(has_norm2_v<std::vector<Vec2>>);

    const std::vector<double> v_scalar{1.0, 2.0, 3.0};
    REQUIRE(traits::norm2(v_scalar) == Catch::Approx(14.0));

    const std::vector<Vec2> v_vec{Vec2(1.0, 2.0), Vec2(3.0, 4.0)};
    REQUIRE(traits::norm2(v_vec) == Catch::Approx(30.0));
  }

  SECTION("std::array") {
    static_assert(has_norm2_v<std::array<double, 3>>);
    static_assert(has_norm2_v<std::array<Vec2, 2>>);

    const std::array<double, 3> a_scalar{2.0, 3.0, 4.0};
    REQUIRE(traits::norm2(a_scalar) == Catch::Approx(29.0));

    const std::array<Vec2, 2> a_vec{Vec2(1.0, 0.0), Vec2(0.0, 2.0)};
    REQUIRE(traits::norm2(a_vec) == Catch::Approx(5.0));
  }

  SECTION("params_pack") {
    ParamsPack pack(Vec2(1.0, 2.0), 3.0, std::vector<double>{4.0, 5.0});
    static_assert(has_norm2_v<decltype(pack)>);
    // (1^2 + 2^2) + 3^2 + (4^2 + 5^2) = 5 + 9 + 41 = 55
    REQUIRE(traits::norm2(pack) == Catch::Approx(55.0));
    REQUIRE(tinyopt::norm2(pack) == Catch::Approx(55.0));
  }

  SECTION("std::pair") {
    const auto p = std::make_pair(Vec2(1.0, 2.0), 3.0);
    static_assert(has_norm2_v<decltype(p)>);
    REQUIRE(traits::norm2(p) == Catch::Approx(14.0));
  }

  SECTION("custom struct and negative check") {
    static_assert(has_norm2_v<CustomWithNorm2>);
    CustomWithNorm2 c;
    REQUIRE(traits::norm2(c) == Catch::Approx(42.0));

    static_assert(!has_norm2_v<CustomWithoutNorm2>);
    static_assert(!has_norm2_v<std::vector<CustomWithoutNorm2>>);
    static_assert(!has_norm2_v<std::array<CustomWithoutNorm2, 2>>);
    static_assert(!has_norm2_v<ParamsPack<Vec2, CustomWithoutNorm2>>);
  }
}
