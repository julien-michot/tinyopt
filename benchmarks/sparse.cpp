// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <cmath>

#if CATCH2_VERSION == 2
#include <catch2/catch.hpp>
#else
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#endif

#include <tinyopt/tinyopt.h>
#include "options.h"

using namespace tinyopt;
using namespace tinyopt::benchmark;
using namespace tinyopt::nlls::lm;

auto simple_loss = [](const auto &x, auto &grad, SparseMatrix<double> &H) {
  const VecX res = 10 * x.array() - 2;
  // Update the gradient and Hessian approx.
  if constexpr (!traits::is_nullptr_v<decltype(grad)>) {
    MatX J = MatX::Zero(res.rows(), x.size());
    for (int i = 0; i < x.size(); ++i) J(i, i) = 10;
    // Update the gradient
    grad = J.transpose() * res;
    // Show various ways to update the H
    if constexpr (0) {
      for (int i = 0; i < x.size(); ++i) H.coeffRef(i, i) = 10 * 10;
      H.makeCompressed();      // Optional
    } else if constexpr (0) {  // Faster update for large matrices
      std::vector<Eigen::Triplet<double>> triplets;
      triplets.reserve(x.size());
      for (int i = 0; i < x.size(); ++i) triplets.emplace_back(i, i, 10 * 10);
      H.setFromTriplets(triplets.begin(), triplets.end());
    } else if constexpr (0) {  // yet another way, using a dense jacobian
      H = (J.transpose() * J).sparseView();
    } else {  // yet another way, using a sparse jacobian
      SparseMatrix<double> Js(res.rows(), x.size());
      for (int i = 0; i < x.size(); ++i) Js.coeffRef(i, i) = 10;
      H = Js.transpose() * Js;
    }
  }
  // Returns the norm + number of residuals
  return Cost(res.norm(), res.size());
};

TEST_CASE("Sparse", "[benchmark][dyn][sparse]") {
  auto dims = GENERATE(10, 100, 1000);
  CAPTURE(dims);

  const Options options = CreateOptions();

  BENCHMARK(std::to_string(dims) + "x" + std::to_string(dims) + " Prior") {
    VecX x = VecX::Random(dims);
    return Optimize(x, simple_loss, options);
  };
}