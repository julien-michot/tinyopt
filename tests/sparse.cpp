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

#if CATCH2_VERSION == 2
#include <catch2/catch.hpp>
#else
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#endif

#include <tinyopt/tinyopt.h>

using namespace tinyopt;
using namespace tinyopt::nlls;

using Catch::Approx;


TEST_CASE("tinyopt_sparse", "[sparse]") {

  auto loss = [&](const auto &x, auto &grad, SparseMatrix<double> &H) {
    const VecX res = 10 * x.array() - 2.0;
    // Update the gradient and Hessian approx.
    if constexpr (!traits::is_nullptr_v<decltype(grad)>) {
      // Define the Jacobian
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

  VecX x = VecX::Random(100);
  nlls::Options options;
  options.check_final_err = false;
  options.log.print_x = false;
  options.log.print_max_stdev = false;
  const auto &out = nlls::Optimize(x, loss, options);
  REQUIRE(out.Succeeded());
  REQUIRE(out.Converged());
  REQUIRE(x.minCoeff() == Approx(0.2).margin(1e-5));
  REQUIRE(x.maxCoeff() == Approx(0.2).margin(1e-5));
 }



TEST_CASE("tinyopt_sparse_ad", "[sparse]") {
  // Optimization using a sparse solver and automatic differentiation

  auto loss = [&](const auto &x) {
    using T = std::decay_t<decltype(x)>::Scalar; // float or Jet
    return (T(10) * x.array() - T(2)).matrix().eval();
  };

  VecXf x = VecXf::Random(10);
  nlls::Options options;
  options.check_final_err = false;
  options.log.print_x = false;
  options.log.print_max_stdev = false;
  using Optimizer = Optimizer<SparseMatrix<float>>;
  Optimizer optimizer(options);
  const auto &out = optimizer(x, loss);

  REQUIRE(out.Succeeded());
  REQUIRE(out.Converged());
  REQUIRE(x.minCoeff() == Approx(0.2).margin(1e-5));
  REQUIRE(x.maxCoeff() == Approx(0.2).margin(1e-5));
 }