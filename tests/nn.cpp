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

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <type_traits>
#include <utility>

#if CATCH2_VERSION == 2
#include <catch2/catch.hpp>
#else
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#endif

#include <tinyopt/diff/auto_diff.h>
#include <tinyopt/diff/num_diff.h>
#include <tinyopt/optimizers/gd.h>

using Catch::Approx;
using namespace tinyopt;

/// @brief Sigmoid = 1/(1+e^-x) and derivative = Sigmoid(x) * (1 - Sigmoid(x))
template <typename T, typename ExportJ = std::nullptr_t>
auto Sigmoid(const T &x, const ExportJ &Jx_or_bool = nullptr) {
  using std::exp;
  constexpr bool HasJac = traits::is_matrix_or_scalar_v<std::decay_t<ExportJ>>;
  constexpr bool IsMatrix = traits::is_matrix_or_array_v<T> || traits::is_sparse_matrix_v<T>;
  if constexpr (traits::is_pair_v<T>) {  // T is a std::pair
    return Sigmoid(x.first, x.second);
  } else {
    if constexpr (std::is_null_pointer_v<ExportJ>) {  // No Jacobian -> return {s}
      if constexpr (IsMatrix) {                       // Matrix -> per element sigmoid
        using Scalar = typename T::Scalar;
        return x.unaryExpr([](Scalar v) { return Sigmoid(v); }).eval();
      } else {  // Scalar
        return T(1.0) / (T(1.0) + exp(-x));
      }
    } else if constexpr (IsMatrix) {
      const auto s = Sigmoid(x);
      const auto Js = s.unaryExpr([](auto v) { return v - v * v; }).reshaped().asDiagonal();
      if constexpr (HasJac) {  // Return {s, Js * Jx}
        return std::make_pair(s, (Js * Jx_or_bool).matrix().eval());
      } else {  // Return {s, Js}
        using Scalar = typename T::Scalar;
        constexpr int DimsJ = traits::params_trait<T>::Dims;
        return std::make_pair(s, Matrix<Scalar, DimsJ, DimsJ>(Js));
      }
    } else {  // Scalar
      const auto s = Sigmoid(x);
      if constexpr (HasJac) {  // Return {s, Js * Jx}
        return std::make_pair(s, ((s - s * s) * Jx_or_bool).matrix().eval());
      } else {  // Return {s, Js}
        return std::make_pair(s, s - s * s);
      }
    }
  }
}

template <typename _Scalar, int N>
struct Perceptron {
  using Scalar = _Scalar;
  using Vec = Vector<Scalar, N>;
  static constexpr Index Dims = N /*weights*/ + 1 /*bias*/;

  Perceptron() {}
  Perceptron(const Vec &_w, Scalar _b) : w{_w}, b{_b} {}

  // Not needed if you use manual Jacobians instead of automatic differentiation
  template <typename T2>
  Perceptron<T2, N> cast() const {
    return Perceptron<T2, N>(w.template cast<T2>(), static_cast<T2>(b));
  }

  /// Define how to update the object members (parametrization and manifold)
  /// Following params()'s order {w,b}
  Perceptron &operator+=(const auto &delta) {
    w += delta.template head<N>();
    b += delta[Dims - 1];
    return *this;
  }

  /// Forward pass / Inference
  auto operator()(const auto &x) const { return Forward(x); }
  auto operator()(const auto &x, const auto &Jx_or_bool) const { return Forward(x, Jx_or_bool); }

  /// Forward + Backward pass -> returns z (and Jacobian)
  /// @arg x a NxB matrix
  /// @arg Jx_or_bool options to return Jy, {nullptr, bool or a a DxM matrix}
  /// @return y:Bx1, Jy: BxD or BxM or skipped
  template <typename Derived, typename ExportJ = std::nullptr_t>
  auto Forward(const MatrixBase<Derived> &x, const ExportJ &Jx_or_bool = nullptr) const {
    return ::Sigmoid(Linear(x, Jx_or_bool));
  }

  /// Linear Layer y = w * x + b,
  /// @arg x NxB matrix
  /// @arg Jx_or_bool options to return Jy, {nullptr, bool or a a DxM matrix}
  /// @return y:Bx1, Jy: BxD or BxM or skipped
  template <typename Derived, typename ExportJ = std::nullptr_t>
  auto Linear(const MatrixBase<Derived> &x, const ExportJ &Jx_or_bool = nullptr) const {
    constexpr int B = Derived::ColsAtCompileTime;  // Batch size
    if constexpr (std::is_null_pointer_v<ExportJ>) {
      if constexpr (B == 1)
        return w.dot(x) + b;
      else
        return ((x.transpose() * w).array() + b).transpose().matrix().eval();
    } else {
      const auto y = Linear(x);                    // Bx1
      Matrix<Scalar, B, Dims> Jy(x.cols(), Dims);  // J: BxD
      Jy.template leftCols<N>() = x.transpose();
      Jy.template rightCols<1>().setOnes();
      if constexpr (traits::is_bool_v<ExportJ>)
        return std::make_pair(y, Jy);
      else {
        return std::make_pair(y, (Jy * Jx_or_bool).eval());
      }
    }
  }

  template <typename X_t, typename J_t>
  auto Linear(const std::pair<X_t, J_t> &pair) const {
    return Linear(pair.first, pair.second);
  }
  template <typename X_t, typename J_t>
  auto Sigmoid(const std::pair<X_t, J_t> &pair) const {
    return ::Sigmoid(pair.first, pair.second);
  }

  /// Return a vector of parameters, params = {w,b},
  Vector<Scalar, Dims> params() const {
    Vector<Scalar, Dims> wb = w.homogeneous();
    wb[Dims - 1] = b;
    return wb;
  }

  Vec w = Vec::Random();   ///< Nx1 Weights
  Scalar b = Scalar(0.0);  ///< 1 Bias
};

void TestPerceptron() {
  constexpr int N = 5;
  using P = Perceptron<float, N>;
  constexpr Index Dims = P::Dims;
  P perceptron;

  const Vector<float, N> sample = Vector<float, N>(1, 2, 3, 4, 5);

  SECTION("Basic") { std::cout << "out:" << perceptron(sample) << "\n"; }

  SECTION("Linear") {
    {
      const auto &[y, Jy] = perceptron.Linear(sample, true);
      REQUIRE(Jy.rows() == 1);
      REQUIRE(Jy.cols() == Dims);
      // Check Jacobian
      auto J = diff::CalculateJac(perceptron, [&](const auto &p) { return p.Linear(sample); });
      REQUIRE(J.rows() == 1);
      REQUIRE(J.cols() == Dims);
      REQUIRE((Jy - J).cwiseAbs().maxCoeff() == Approx(0.0).margin(1e-5));
    }
    constexpr int B = 3;                                               // batch size
    const Matrix<float, N, B> batch2 = Matrix<float, N, B>::Random();  // NxB
    {
      auto Jad = diff::CalculateJac(perceptron, [&](const auto &p) {
        using T = std::decay_t<decltype(p)>::Scalar;
        const auto b = batch2.template cast<T>().eval();
        return p.Linear(b);
      });
      const auto &[y, Jy] = perceptron.Linear(batch2, true);
      REQUIRE((Jy - Jad).cwiseAbs().maxCoeff() == Approx(0.0).margin(1e-5));
    }
    // Check a chained rule
    {
      constexpr int W = 4;
      Vec4f V(3, 2, 1, 0);

      const auto J = diff::CalculateJac(V, [&](const auto &v) {
        using T = std::decay_t<decltype(v[0])>;
        Vector<T, N + 1> u;
        u[0] = v[0] + T(2) * v[1];
        u[1] = v[1];
        u[2] = v[2];
        u[3] = v[3];
        u[4] = v[1] + v[2];
        u[5] = T(0.0 * B);
        Perceptron<T, N> p;
        p.w.setZero();
        p.b = T(0);
        p += u;
        if constexpr (1) {
          const auto b = batch2.template cast<T>().eval();
          return p.Linear(b);  // Bx1
        } else {
          return p.params();  // includes manifold
        }
      });
      REQUIRE(J.rows() == B);
      REQUIRE(J.cols() == W);

      Matrix<float, N + 1, 4> Jv;  // dP/dV = Jv: DxW
      Jv << 1, 2, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0;

      const auto &[y, Jy] = perceptron.Linear(batch2, Jv);
      REQUIRE(Jy.rows() == B);
      REQUIRE(Jy.cols() == W);
      REQUIRE((Jy - J).cwiseAbs().maxCoeff() == Approx(0.0).margin(1e-5));
    }
  }

  SECTION("Sigmoid") {
    constexpr int B = 3;  // batch size
    const Matrix<float, N, B> batch2 = Matrix<float, N, B>::Random();
    {
      const auto &[z, Jz] = Sigmoid(sample, true);
      REQUIRE(z.size() == N);
      REQUIRE(Jz.rows() == N);
      REQUIRE(Jz.cols() == N);
    }
    {
      const auto &[z, Jz] = Sigmoid(batch2, true);
      const auto J = diff::CalculateJac(batch2, [&](const auto &b) { return Sigmoid(b); });
      REQUIRE(z.size() == N * B);
      REQUIRE(Jz.rows() == J.rows());
      REQUIRE(Jz.cols() == J.cols());
      REQUIRE((Jz - J).cwiseAbs().maxCoeff() == Approx(0.0).margin(1e-5));
    }
    {
      constexpr int B2 = 2;
      const MatXf batch = MatXf::Random(N, B2);

      const auto J = diff::CalculateJac(perceptron, [&](const auto &p) {
        using T = std::decay_t<decltype(p)>::Scalar;
        const auto b = batch.template cast<T>().eval();
        return p(b);
      });
      REQUIRE(J.rows() == B2);
      REQUIRE(J.cols() == Dims);

      const auto &[z, Jz] = perceptron.Forward(batch, true);
      REQUIRE(z.size() == B2);
      REQUIRE(Jz.rows() == B2);
      REQUIRE(Jz.cols() == Dims);
      REQUIRE((Jz - J).cwiseAbs().maxCoeff() == Approx(0.0).margin(1e-5));
    }
  }

  SECTION("Train One Step with Manual Jacs") {
    constexpr int B = 6;
    const MatXf batch = MatXf::Random(N, B);
    const float scale = 1.0f;

    auto loss = [scale](const auto &z) {
      const VecXf res = scale * z.array() - 0.5f;
      const MatXf J = VecXf::Constant(z.size(), 1, scale).asDiagonal();
      return std::make_pair(res, J);
    };

    // Cost with manual gradient update
    auto cost = [&](const auto &p, auto &grad) {
      if constexpr (!traits::is_nullptr_v<decltype(grad)>) {
        const auto &[z, Jz] = p.Forward(batch, true);
        const auto &[res, Jl] = loss(z);
        const auto J = (Jl * Jz).eval();
        grad = J.transpose() * res;
        return res.norm();
      } else {
        return loss(p.Forward(batch)).first.norm();
      }
    };

    P perceptron2 = perceptron;  // make a copy

    // Optimize with Manual accumulation
    gd::Options options;
    options.solver.lr = 0.1f;
    options.max_iters = 1;
    options.log.print_J_jet = false;
    const auto &out1 = gd::Optimize(perceptron, cost, options);

    // Cost with automatic gradient update
    auto cost2 = [scale, &batch](const auto &p) {
      using T = std::decay_t<decltype(p)>::Scalar;
      const auto b = batch.template cast<T>().eval();
      const auto z = p(b);
      return (T(scale) * z.array() - T(0.5f)).matrix().norm();
    };

    // Optimize with AD
    const auto &out2 = gd::Optimize(perceptron2, cost2, options);

    REQUIRE(std::abs(out1.final_cost.cost - out2.final_cost.cost) == Approx(0).margin(1e-5));
  }
}

TEST_CASE("tinyopt_mlp") { TestPerceptron(); }