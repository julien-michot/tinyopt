// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <tinyopt/losses/robust_norms.h>
#include <tinyopt/params_wrapper.h>
#include <tinyopt/tinyopt.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <vector>

using Catch::Approx;
using namespace tinyopt;

template <typename T>
struct Landmark2D {
  using Scalar = T;
  static constexpr Index Dims = 2;

  Vector<T, 2> position = Vector<T, 2>::Zero();

  Landmark2D &operator+=(const Vector<T, Dims> &delta) {
    position += delta;
    return *this;
  }

  template <typename T2>
  Landmark2D<T2> cast() const {
    return {position.template cast<T2>()};
  }
};

template <typename T>
struct Pose2 {
  using Scalar = T;
  using Vec2 = Vector<T, 2>;
  using Tangent = Vector<T, 3>;
  static constexpr Index Dims = 3;

  Pose2() : translation(Vec2::Zero()), orientation(Vec2(1, 0)) {}

  template <typename T2>
  static auto cast(const Pose2 &pose) {
    Pose2<T2> result;
    result.translation = pose.translation.template cast<T2>();
    result.orientation = pose.orientation.template cast<T2>();
    return result;
  }

  Pose2 &operator+=(const Tangent &delta) {
    translation += delta.template head<2>();

    const T tangent = delta[2];
    const T denominator = T(1) + tangent * tangent;
    const T cosine = (T(1) - tangent * tangent) / denominator;
    const T sine = T(2) * tangent / denominator;
    const Vec2 previous_orientation = orientation;
    orientation[0] = cosine * previous_orientation[0] - sine * previous_orientation[1];
    orientation[1] = sine * previous_orientation[0] + cosine * previous_orientation[1];
    return *this;
  }

  Vec2 translation;
  Vec2 orientation;
};

namespace tinyopt::traits {

template <typename T>
struct params_trait<Pose2<T>> {
  using Scalar = T;
  static constexpr Index Dims = 3;

  template <typename T2>
  static Pose2<T2> cast(const Pose2<T> &pose) {
    Pose2<T2> result;
    result.translation = pose.translation.template cast<T2>();
    result.orientation = pose.orientation.template cast<T2>();
    return result;
  }

  static void PlusEq(Pose2<T> &pose, const Vector<T, Dims> &delta) {
    pose.translation += delta.template head<2>();
    const T tangent = delta[2];
    const T denominator = T(1) + tangent * tangent;
    const T cosine = (T(1) - tangent * tangent) / denominator;
    const T sine = T(2) * tangent / denominator;
    const auto previous_orientation = pose.orientation;
    pose.orientation[0] = cosine * previous_orientation[0] - sine * previous_orientation[1];
    pose.orientation[1] = sine * previous_orientation[0] + cosine * previous_orientation[1];
  }
};

}  // namespace tinyopt::traits

template <typename T = double>
struct TutorialLockedParams {
  using Scalar = T;
  static constexpr Index Dims = 3;
  Vector<T, 3> values{T(0.0), T(5.0), T(0.0)};
  std::vector<Index> locked() const { return {1}; }  // Index 1 is fixed at 5.0
  template <typename DeltaType>
  TutorialLockedParams &operator+=(const DeltaType &delta) {
    values += delta;
    return *this;
  }
  template <typename T2>
  TutorialLockedParams<T2> cast() const {
    TutorialLockedParams<T2> p;
    p.values = values.template cast<T2>();
    return p;
  }
};

struct TutorialSchurParams {
  using Scalar = double;
  static constexpr Index Dims = 4;
  Vector<double, 4> values{0.0, 0.0, 0.0, 0.0};
  std::pair<Index, Index> reduced() const { return {0, 2}; }  // 2 reduced head parameters
  template <typename DeltaType>
  TutorialSchurParams &operator+=(const DeltaType &delta) {
    values += delta;
    return *this;
  }
  template <typename T2>
  TutorialSchurParams cast() const {
    return *this;
  }
};

TEST_CASE("tinyopt_tutorial_compile_examples") {
  Vec2f fixed_point = Vec2f::Zero();
  auto fixed_summary =
      tinyopt::Optimize(fixed_point, [](const auto &x) { return (x - Vec2f(1.0f, 2.0f)).eval(); });
  REQUIRE(fixed_summary.Succeeded());

  std::vector<float> dynamic_values(3, 0.0f);
  auto vector_summary =
      tinyopt::Optimize(dynamic_values, [](const auto &x) { return x[0] + x[1] + x[2] - 3.0f; });
  REQUIRE(vector_summary.Succeeded());

  std::array<Vec3, 2> landmarks{Vec3::Zero(), Vec3::Ones()};
  static_assert(traits::params_trait<decltype(landmarks)>::Dims == 6);
  traits::params_trait<decltype(landmarks)>::PlusEq(landmarks, Vector<double, 6>::Ones());

  double scalar_x = 4.0;
  auto scalar_objective = [](const auto &x) { return (x - 2.0) * (x - 2.0); };
  auto scalar_summary = tinyopt::Optimize(scalar_x, scalar_objective);
  REQUIRE(scalar_summary.Succeeded());

  Vec2 residual_x = Vec2::Zero();
  const Vec2 residual_target(3.0, -2.0);
  auto residuals = [&](const auto &value) { return (value - residual_target).eval(); };
  auto residual_summary = tinyopt::Optimize(residual_x, residuals);
  REQUIRE(residual_summary.Succeeded());

  auto loss = [&](const auto &value, auto &grad, auto &hessian) {
    const Vec2 residual = value - residual_target;
    const Mat2 J = Mat2::Identity();
    if constexpr (!traits::is_nullptr_v<decltype(grad)>) {
      grad.noalias() = J.transpose() * residual;
    }
    if constexpr (!traits::is_nullptr_v<decltype(hessian)>) {
      hessian.noalias() = J.transpose() * J;
    }
    return 0.5 * residual.squaredNorm();
  };
  auto optimizer = lm::Optimizer<Mat2>{};
  const auto manual_summary = optimizer(residual_x, loss);
  REQUIRE(manual_summary.Succeeded());

  Options svd_options;
  svd_options.linear_solver = LinearSolverMethod::SVD;
  svd_options.svd_relative_threshold = 1e-8;
  svd_options.exp.save_history = false;
  svd_options.exp.measure_time = false;
  REQUIRE(svd_options.linear_solver == LinearSolverMethod::SVD);
  REQUIRE(svd_options.svd_relative_threshold == Approx(1e-8));
  REQUIRE_FALSE(svd_options.exp.save_history);
  REQUIRE_FALSE(svd_options.exp.measure_time);

  Vec2 center(5.0, 1.0);
  Vec2 scale(2.0, 3.0);
  auto multi_summary = tinyopt::Optimize(
      center, scale, [&](const auto &c, const auto &s) { return (c - s).eval(); });
  REQUIRE(multi_summary.Succeeded());

  Landmark2D<double> landmark;
  auto custom_summary = tinyopt::Optimize(
      landmark, [](const auto &value) { return (value.position - Vec2(1.0, 2.0)).eval(); });
  REQUIRE(custom_summary.Succeeded());

  Pose2<double> pose;
  ParamsWrapper<Pose2<double> &> wrapped_pose(pose);
  auto wrapped_summary = tinyopt::Optimize(wrapped_pose, [&](const auto &p) {
    Vector<typename std::decay_t<decltype(p.params)>::Scalar, 4> residuals;
    residuals.template head<2>() = p.params.translation - Vec2(2.0, -1.0);
    residuals.template tail<2>() = p.params.orientation - Vec2(0.8, 0.6);
    return residuals;
  });
  REQUIRE(wrapped_summary.Succeeded());
  REQUIRE((pose.translation - Vec2(2.0, -1.0)).norm() == Approx(0.0).margin(1e-5));
  REQUIRE((pose.orientation - Vec2(0.8, 0.6)).norm() == Approx(0.0).margin(1e-5));

  Vec2 robust_x = Vec2::Zero();
  auto robust_summary = tinyopt::Optimize(robust_x, [&](const auto &value) {
    const auto residual = (value - Vec2(1.0, 1.0)).eval();
    return losses::Huber(residual.squaredNorm(), 0.7);
  });
  REQUIRE(robust_summary.Succeeded());

  VecX sparse_x = VecX::Constant(5, 1.0);
  const VecX sparse_target = VecX::Constant(5, 3.0);
  auto sparse_loss = [&](auto &x, auto &grad, SparseMat &hessian) {
    const VecX res = x - sparse_target;
    if constexpr (!traits::is_nullptr_v<decltype(grad)>) {
      grad = res;
      std::vector<Eigen::Triplet<double>> triplets;
      triplets.reserve(x.size());
      for (int i = 0; i < x.size(); ++i) {
        triplets.emplace_back(i, i, 1.0);
      }
      hessian.setFromTriplets(triplets.begin(), triplets.end());
    }
    return 0.5 * res.squaredNorm();
  };
  auto sparse_summary = tinyopt::Optimize(sparse_x, sparse_loss);
  REQUIRE(sparse_summary.Succeeded());
  REQUIRE(sparse_summary.Converged());
  REQUIRE((sparse_x - sparse_target).norm() == Approx(0.0).margin(1e-5));

  using BlockHessian = BlockSparseMatrix<double, 2>;
  Vec4 block_sparse_x(8.0, -6.0, 4.0, -5.0);
  const Vec4 block_sparse_target(2.0, -1.0, 0.5, 3.0);
  const Mat2 first_block{{2.0, 0.3}, {0.3, 4.0}};
  const Mat2 second_block{{3.0, 0.2}, {0.2, 5.0}};
  auto block_sparse_loss = [&](const Vec4 &value, VecX &gradient, BlockHessian &hessian) {
    const Vec4 difference = value - block_sparse_target;
    gradient.head<2>().noalias() = first_block * difference.head<2>();
    gradient.tail<2>().noalias() = second_block * difference.tail<2>();
    hessian.block<2, 2>(0, 0) = first_block;
    hessian.block<2, 2>(1, 1) = second_block;
    return 0.5 * difference.squaredNorm();
  };
  auto block_sparse_summary = tinyopt::Optimize(block_sparse_x, block_sparse_loss);
  REQUIRE(block_sparse_summary.Succeeded());
  REQUIRE(block_sparse_summary.Converged());
  REQUIRE((block_sparse_x - block_sparse_target).norm() == Approx(0.0).margin(1e-5));

  // Locked parameters tutorial example
  TutorialLockedParams<double> locked_p;
  const Vec3 target_3d{1.0, 2.0, 3.0};
  auto locked_summary = tinyopt::Optimize(locked_p, [&](const auto &p) {
    return (p.values - target_3d.template cast<typename std::decay_t<decltype(p)>::Scalar>())
        .eval();
  });
  REQUIRE(locked_summary.Succeeded());
  REQUIRE(locked_p.values[0] == Approx(target_3d[0]).margin(1e-5));
  REQUIRE(locked_p.values[1] == Approx(5.0));  // Remains unchanged
  REQUIRE(locked_p.values[2] == Approx(target_3d[2]).margin(1e-5));

  // 1D scalar locked tutorial example
  float scalar_locked_x = 3.0f;
  gn::Optimizer<Matrix<float, 1, 1>> scalar_locked_opt;
  scalar_locked_opt.lock({0});
  auto scalar_locked_summary =
      scalar_locked_opt.Optimize(scalar_locked_x, [](const auto &val, auto &g, auto &H) {
        const float diff = val - 10.0f;
        if constexpr (!traits::is_nullptr_v<decltype(g)>) {
          g(0) = diff;
          H(0, 0) = 1.0f;
        }
        return 0.5f * diff * diff;
      });
  REQUIRE(scalar_locked_summary.Succeeded());
  REQUIRE(scalar_locked_x == Approx(3.0f));

  // Sparse Schur complement tutorial example
  TutorialSchurParams schur_p;
  const Vector<double, 4> schur_target{1.0, 2.0, 3.0, 4.0};
  auto schur_loss = [&](const auto &p, auto &grad, SparseMat &hessian) {
    const Vector<double, 4> diff = p.values - schur_target;
    if constexpr (!traits::is_nullptr_v<decltype(grad)>) {
      std::vector<Eigen::Triplet<double>> triplets;
      for (int i = 0; i < 4; ++i) triplets.emplace_back(i, i, 2.0);
      triplets.emplace_back(0, 2, 0.5);
      triplets.emplace_back(2, 0, 0.5);
      hessian.resize(4, 4);
      hessian.setFromTriplets(triplets.begin(), triplets.end());
      grad = hessian * diff;
    }
    return 0.5 * diff.squaredNorm();
  };
  gn::Optimizer<SparseMat> schur_opt;
  auto schur_summary = schur_opt.Optimize(schur_p, schur_loss);
  REQUIRE(schur_summary.Succeeded());
  REQUIRE(schur_opt.hasReduced());
  REQUIRE(schur_opt.numReduced() == 2);
  REQUIRE((schur_p.values - schur_target).norm() == Approx(0.0).margin(1e-5));
}
