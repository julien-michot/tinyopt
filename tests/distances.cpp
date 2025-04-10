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

#include <iostream>
#include "tinyopt/math.h"

#if CATCH2_VERSION == 2
#include <catch2/catch.hpp>
#else
#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#endif

#include <tinyopt/diff/num_diff.h>
#include <tinyopt/distances.h>
#include <tinyopt/log.h>

using Catch::Approx;
using namespace tinyopt;
using namespace tinyopt::distances;
using namespace tinyopt::diff;

#define LOSS_WRAPPER(name, func, expected_code)                      \
  struct name {                                                      \
    name() {}                                                        \
    template <typename T>                                            \
    auto operator()(const T &a, const T &b) const {                  \
      return func(a, b);                                             \
    }                                                                \
    template <typename T>                                            \
    auto operator()(const T &a, const T &b, bool export_jac) const { \
      return func(a, b, export_jac);                                 \
    }                                                                \
    template <typename T>                                            \
    auto expected(const T &a, const T &b) const {                    \
      if constexpr (traits::is_matrix_or_array_v<T>)                 \
        return ((expected_code));                                    \
      else                                                           \
        return expected(Vector<T, 1>(a), Vector<T, 1>(b));           \
    }                                                                \
  };

LOSS_WRAPPER(L1Wrapper, L1, (a - b).template lpNorm<1>());
LOSS_WRAPPER(L2Wrapper, L2, (a - b).norm());
LOSS_WRAPPER(LInfWrapper, Linf, (a - b).template lpNorm<Infinity>());
LOSS_WRAPPER(CosineWrapper, Cosine, a.normalized().dot(b).normalized());

TEMPLATE_TEST_CASE("tinyopt_distances", "[loss]", L1Wrapper, L2Wrapper) {
  TestType dist;
  SECTION("Scalar") {
    float a = 0.8f, b = 0.3f;
    const auto &[d, Ja, Jb] = dist(a, b, true);
    TINYOPT_LOG("dist = [{}, Ja:{}, Jb:{}] exp:{}", d, Ja, Jb, dist.expected(a, b));
    REQUIRE(d == Approx(dist.expected(a, b)).margin(1e-5));

    const auto Ja_num = EstimateNumJac(a, [&](const auto &a) { return dist(a, b); });
    const auto Jb_num = EstimateNumJac(b, [&](const auto &b) { return dist(a, b); });
    TINYOPT_LOG("Ja_num:{}", Ja_num);
    TINYOPT_LOG("Jb_num:{}", Jb_num);
    REQUIRE(std::abs(Ja - Ja_num[0]) == Approx(0.0).margin(1e-3));
    REQUIRE(std::abs(Jb - Jb_num[0]) == Approx(0.0).margin(1e-3));
  }
  SECTION("Vec4 + Jac") {
    Vec4 a = Vec4::Random(), b = Vec4::Random();
    const auto &[d, Ja, Jb] = dist(a, b, true);
    TINYOPT_LOG("dist = [{}, Ja:{}, Jb:{}] exp:{}", d, Ja, Jb, dist.expected(a, b));
    REQUIRE(d == Approx(dist.expected(a, b)).margin(1e-5));

    const auto Ja_num = EstimateNumJac(a, [&](const auto &a) { return dist(a, b); });
    const auto Jb_num = EstimateNumJac(b, [&](const auto &b) { return dist(a, b); });
    TINYOPT_LOG("Ja_num:{}", Ja_num);
    TINYOPT_LOG("Jb_num:{}", Jb_num);
    REQUIRE((Ja - Ja_num).cwiseAbs().maxCoeff() == Approx(0.0).margin(1e-3));
    REQUIRE((Jb - Jb_num).cwiseAbs().maxCoeff() == Approx(0.0).margin(1e-3));
  }
}

template <typename Scalar, int Dims>
auto CreateCov(int dims = Dims, Scalar eps = 1e-2) {
  using Mat = Matrix<Scalar, Dims, Dims>;
  Mat A = Mat::Random(dims, dims);
  Mat C = A * A.transpose() + eps * Mat::Identity();
  return C;
}

TEST_CASE("tinyopt_maha_distances", "[loss]") {
  SECTION("Scalar") {
    float a = 0.8f, b = 0.3f, var = 2.0;
    const auto &[d, Ja, Jb] = MahaNorm(a, b, var, true);
    const auto expected_norm = std::sqrt((a - b) * (a - b) / var);
    TINYOPT_LOG("dist = [{}, Ja:{}, Jb:{}] exp:{}", d, Ja, Jb, expected_norm);
    REQUIRE(d == Approx(expected_norm).margin(1e-5));

    const auto Ja_num = EstimateNumJac(a, [&](const auto &a) { return MahaNorm(a, b, var); });
    const auto Jb_num = EstimateNumJac(b, [&](const auto &b) { return MahaNorm(a, b, var); });
    TINYOPT_LOG("Ja_num:{}", Ja_num);
    TINYOPT_LOG("Jb_num:{}", Jb_num);
    REQUIRE(std::abs(Ja - Ja_num[0]) == Approx(0.0).margin(1e-3));
    REQUIRE(std::abs(Jb - Jb_num[0]) == Approx(0.0).margin(1e-3));
  }
  SECTION("Vec4 + Jac") {
    Vec4 a = Vec4::Random(), b = Vec4::Random();
    const auto cov = CreateCov<double, 4>();
    const auto &[d, Ja, Jb] = MahaNorm(a, b, cov, true);
    const auto expected_norm = std::sqrt((a - b).transpose() * cov.inverse() * (a - b));
    TINYOPT_LOG("dist = [{}, Ja:{}, Jb:{}] exp:{}", d, Ja, Jb, expected_norm);
    REQUIRE(d == Approx(expected_norm).margin(1e-5));

    const auto Ja_num = EstimateNumJac(a, [&](const auto &a) { return MahaNorm(a, b, cov); });
    const auto Jb_num = EstimateNumJac(b, [&](const auto &b) { return MahaNorm(a, b, cov); });
    TINYOPT_LOG("Ja_num:{}", Ja_num);
    TINYOPT_LOG("Jb_num:{}", Jb_num);
    REQUIRE((Ja - Ja_num).cwiseAbs().maxCoeff() == Approx(0.0).margin(1e-3));
    REQUIRE((Jb - Jb_num).cwiseAbs().maxCoeff() == Approx(0.0).margin(1e-3));
  }
}
