// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cmath>
#include <vector>

#if CATCH2_VERSION == 2
#include <catch2/catch.hpp>
#else
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#endif

#include <tinyopt/diff/gradient_check.h>
#include <tinyopt/tinyopt.h>

using Catch::Approx;
using namespace tinyopt;

namespace {

// Rectangle storing 4 points (Vector<T, 2> pts[4])
template <typename T = float>
struct Rectangle {
  using Scalar = T;
  using Vec2 = Vector<T, 2>;
  static constexpr Index Dims = 8;

  Vec2 pts[4];
  int fixed_point = -1;  // -1: none, 0: first, 1: second, 3: last

  Rectangle() {
    for (int i = 0; i < 4; ++i) pts[i] = Vec2::Zero();
  }

  Rectangle(const Vec2 &p0, const Vec2 &p1, const Vec2 &p2, const Vec2 &p3, int fixed = -1)
      : fixed_point(fixed) {
    pts[0] = p0;
    pts[1] = p1;
    pts[2] = p2;
    pts[3] = p3;
  }

  // User-defined locked method returning indices of fixed parameters
  std::array<Index, 2> locked() const {
    if (fixed_point >= 0 && fixed_point < 4) {
      return {fixed_point * 2, fixed_point * 2 + 1};
    }
    return {};
  }

  template <typename T2>
  Rectangle<T2> cast() const {
    Rectangle<T2> r;
    r.fixed_point = fixed_point;
    for (int i = 0; i < 4; ++i) {
      r.pts[i] = pts[i].template cast<T2>();
    }
    return r;
  }

  template <typename T2>
  static Rectangle<T2> cast(const Rectangle<T> &rect) {
    return rect.template cast<T2>();
  }

  template <typename DeltaType>
  Rectangle &operator+=(const DeltaType &delta) {
    for (int i = 0; i < 4; ++i) {
      pts[i] += delta.template segment<2>(i * 2);
    }
    return *this;
  }
};

// Rectangle without any locked() trait or method
template <typename T = float>
struct RectangleNoTrait {
  using Scalar = T;
  using Vec2 = Vector<T, 2>;
  static constexpr Index Dims = 8;

  Vec2 pts[4];

  RectangleNoTrait() {
    for (int i = 0; i < 4; ++i) pts[i] = Vec2::Zero();
  }

  RectangleNoTrait(const Vec2 &p0, const Vec2 &p1, const Vec2 &p2, const Vec2 &p3) {
    pts[0] = p0;
    pts[1] = p1;
    pts[2] = p2;
    pts[3] = p3;
  }

  template <typename T2>
  RectangleNoTrait<T2> cast() const {
    RectangleNoTrait<T2> r;
    for (int i = 0; i < 4; ++i) {
      r.pts[i] = pts[i].template cast<T2>();
    }
    return r;
  }

  template <typename T2>
  static RectangleNoTrait<T2> cast(const RectangleNoTrait<T> &rect) {
    return rect.template cast<T2>();
  }

  template <typename DeltaType>
  RectangleNoTrait &operator+=(const DeltaType &delta) {
    for (int i = 0; i < 4; ++i) {
      pts[i] += delta.template segment<2>(i * 2);
    }
    return *this;
  }
};

struct WobblyRectangleData {
  Vec2f base_pts[4];
  Vec2f wobbly_pts[4];
  float target_width = 4.0f;
  float target_height = 2.0f;

  WobblyRectangleData() {
    base_pts[0] = Vec2f(-2.0f, -1.0f);
    base_pts[1] = Vec2f(2.0f, -1.0f);
    base_pts[2] = Vec2f(2.0f, 1.0f);
    base_pts[3] = Vec2f(-2.0f, 1.0f);

    // Distorted/wobbly points
    wobbly_pts[0] = Vec2f(-2.15f, -1.10f);
    wobbly_pts[1] = Vec2f(2.25f, -0.85f);
    wobbly_pts[2] = Vec2f(1.85f, 1.20f);
    wobbly_pts[3] = Vec2f(-1.90f, 0.90f);
  }

  Rectangle<float> CreateInitialRectangle(int fixed_point = -1) const {
    Vec2f pts[4];
    for (int i = 0; i < 4; ++i) pts[i] = wobbly_pts[i];
    if (fixed_point >= 0 && fixed_point < 4) {
      pts[fixed_point] = base_pts[fixed_point];
    }
    return Rectangle<float>(pts[0], pts[1], pts[2], pts[3], fixed_point);
  }

  RectangleNoTrait<float> CreateInitialRectangleNoTrait() const {
    return RectangleNoTrait<float>(base_pts[0], wobbly_pts[1], wobbly_pts[2], wobbly_pts[3]);
  }
};

// Automatic differentiation loss function with 1 input argument: loss(r)
// Optimizes a wobbly rectangle with soft constraints on width (4.0) and height (2.0)
auto MakeWobblyRectangleLoss() {
  return [](const auto &r) {
    using Scalar = typename std::decay_t<decltype(r)>::Scalar;
    const auto &p0 = r.pts[0];
    const auto &p1 = r.pts[1];
    const auto &p2 = r.pts[2];
    const auto &p3 = r.pts[3];

    const Scalar target_w(4.0);
    const Scalar target_h(2.0);

    const auto edge_bottom = p1 - p0;
    const auto edge_top = p2 - p3;
    const auto edge_left = p3 - p0;
    const auto edge_right = p2 - p1;

    // Soft constraints on rectangle width
    const auto dw_bottom = edge_bottom.norm() - target_w;
    const auto dw_top = edge_top.norm() - target_w;

    // Soft constraints on rectangle height
    const auto dh_left = edge_left.norm() - target_h;
    const auto dh_right = edge_right.norm() - target_h;

    // Rectangle shape constraints:
    // Parallel opposite edges
    const auto par_w = edge_bottom - edge_top;
    const auto par_h = edge_left - edge_right;

    // Orthogonal adjacent edges: (p1 - p0) . (p3 - p0) == 0
    const auto ortho = edge_bottom.dot(edge_left) / (target_w * target_h);

    // Horizontal bottom edge alignment
    const auto horiz = edge_bottom.y();

    return dw_bottom * dw_bottom + dw_top * dw_top + dh_left * dh_left + dh_right * dh_right +
           par_w.squaredNorm() + par_h.squaredNorm() + ortho * ortho + horiz * horiz;
  };
}

}  // namespace

TEST_CASE("tinyopt_locked_traits") {
  SECTION("Default locked traits for basic types") {
    REQUIRE_FALSE(traits::has_locked_v<float>);
    REQUIRE_FALSE(traits::has_locked_v<Vec3>);
    REQUIRE_FALSE(traits::has_locked_v<std::vector<double>>);
    REQUIRE_FALSE(traits::has_locked_v<std::array<Vec2, 2>>);
    REQUIRE_FALSE(traits::has_locked_v<RectangleNoTrait<float>>);
  }

  SECTION("User defined Rectangle locked") {
    REQUIRE(traits::has_locked_v<Rectangle<float>>);
    Rectangle rect;
    rect.fixed_point = 0;
    auto locked_indices = traits::locked(rect);
    REQUIRE(locked_indices.size() == 2);
    REQUIRE(locked_indices[0] == 0);
    REQUIRE(locked_indices[1] == 1);

    rect.fixed_point = 3;
    locked_indices = traits::locked(rect);
    REQUIRE(locked_indices.size() == 2);
    REQUIRE(locked_indices[0] == 6);
    REQUIRE(locked_indices[1] == 7);
  }
}

TEST_CASE("tinyopt_conflict_locked_trait_and_set_locked") {
  WobblyRectangleData data;
  Rectangle<float> rect = data.CreateInitialRectangle(0);  // has trait locked
  auto loss = MakeWobblyRectangleLoss();

  Options options(Options::Solver::GradientDescent);
  gd::Optimizer<Vector<float, 8>> opt(options);
  opt.lock({2, 3});  // user also specifies locked parameters

  REQUIRE_THROWS_AS(opt.Optimize(rect, loss), std::invalid_argument);
}

TEST_CASE("tinyopt_rectangle_gradient_check") {
  WobblyRectangleData data;
  Rectangle<float> rect = data.CreateInitialRectangle();
  auto loss = MakeWobblyRectangleLoss();

  // Verify autodiff derivatives of the 1-arg loss function against numerical gradient
  auto acc = [&](const Rectangle<float> &r, auto &grad) -> float {
    const auto [val, J] = diff::Eval(r, loss);
    if constexpr (!traits::is_nullptr_v<decltype(grad)>) {
      grad = J;
    }
    return val;
  };

  REQUIRE(diff::CheckGradient(rect, acc));
}

TEST_CASE("tinyopt_rectangle_fixing_first_point") {
  WobblyRectangleData data;
  Rectangle<float> rect = data.CreateInitialRectangle(0);  // Fix first point (pts[0])
  const Rectangle<float> initial_rect = rect;
  auto loss = MakeWobblyRectangleLoss();

  // Initial wobbly rectangle violates constraints
  REQUIRE(std::abs((initial_rect.pts[1] - initial_rect.pts[0]).norm() - data.target_width) > 0.1f);

  Options options(Options::Solver::GradientDescent);
  options.gd.lr = 0.05f;
  options.stop.max_iters = 200;
  options.stop.min_grad_norm2 = 1e-8f;
  options.stop.min_step_norm2 = 1e-8f;

  const auto sum = Optimize(rect, loss, options);
  REQUIRE(sum.Succeeded());

  // First point must be strictly fixed/locked
  REQUIRE(rect.pts[0].x() == Approx(initial_rect.pts[0].x()).margin(1e-6f));
  REQUIRE(rect.pts[0].y() == Approx(initial_rect.pts[0].y()).margin(1e-6f));

  // Other points converged to nominal rectangle with soft width/height constraints satisfied
  REQUIRE((rect.pts[1] - data.base_pts[1]).norm() == Approx(0.0f).margin(1e-2f));
  REQUIRE((rect.pts[2] - data.base_pts[2]).norm() == Approx(0.0f).margin(1e-2f));
  REQUIRE((rect.pts[3] - data.base_pts[3]).norm() == Approx(0.0f).margin(1e-2f));

  // Width and height constraints strictly met
  REQUIRE((rect.pts[1] - rect.pts[0]).norm() == Approx(data.target_width).margin(1e-2f));
  REQUIRE((rect.pts[2] - rect.pts[3]).norm() == Approx(data.target_width).margin(1e-2f));
  REQUIRE((rect.pts[3] - rect.pts[0]).norm() == Approx(data.target_height).margin(1e-2f));
  REQUIRE((rect.pts[2] - rect.pts[1]).norm() == Approx(data.target_height).margin(1e-2f));
}

TEST_CASE("tinyopt_rectangle_fixing_second_point") {
  WobblyRectangleData data;
  Rectangle<float> rect = data.CreateInitialRectangle(1);  // Fix second point (pts[1])
  const Rectangle<float> initial_rect = rect;
  auto loss = MakeWobblyRectangleLoss();

  Options options(Options::Solver::GradientDescent);
  options.gd.lr = 0.05f;
  options.stop.max_iters = 200;
  options.stop.min_grad_norm2 = 1e-8f;
  options.stop.min_step_norm2 = 1e-8f;

  const auto sum = Optimize(rect, loss, options);
  REQUIRE(sum.Succeeded());

  // Second point must be strictly fixed/locked
  REQUIRE(rect.pts[1].x() == Approx(initial_rect.pts[1].x()).margin(1e-6f));
  REQUIRE(rect.pts[1].y() == Approx(initial_rect.pts[1].y()).margin(1e-6f));

  // Other points converged to nominal rectangle
  REQUIRE((rect.pts[0] - data.base_pts[0]).norm() == Approx(0.0f).margin(1e-2f));
  REQUIRE((rect.pts[2] - data.base_pts[2]).norm() == Approx(0.0f).margin(1e-2f));
  REQUIRE((rect.pts[3] - data.base_pts[3]).norm() == Approx(0.0f).margin(1e-2f));

  // Width and height constraints strictly met
  REQUIRE((rect.pts[1] - rect.pts[0]).norm() == Approx(data.target_width).margin(1e-2f));
  REQUIRE((rect.pts[2] - rect.pts[3]).norm() == Approx(data.target_width).margin(1e-2f));
  REQUIRE((rect.pts[3] - rect.pts[0]).norm() == Approx(data.target_height).margin(1e-2f));
  REQUIRE((rect.pts[2] - rect.pts[1]).norm() == Approx(data.target_height).margin(1e-2f));
}

TEST_CASE("tinyopt_rectangle_fixing_last_point") {
  WobblyRectangleData data;
  Rectangle<float> rect = data.CreateInitialRectangle(3);  // Fix last point (pts[3])
  const Rectangle<float> initial_rect = rect;
  auto loss = MakeWobblyRectangleLoss();

  Options options(Options::Solver::GradientDescent);
  options.gd.lr = 0.05f;
  options.stop.max_iters = 200;
  options.stop.min_grad_norm2 = 1e-8f;
  options.stop.min_step_norm2 = 1e-8f;

  const auto sum = Optimize(rect, loss, options);
  REQUIRE(sum.Succeeded());

  // Last point must be strictly fixed/locked
  REQUIRE(rect.pts[3].x() == Approx(initial_rect.pts[3].x()).margin(1e-6f));
  REQUIRE(rect.pts[3].y() == Approx(initial_rect.pts[3].y()).margin(1e-6f));

  // Other points converged to nominal rectangle
  REQUIRE((rect.pts[0] - data.base_pts[0]).norm() == Approx(0.0f).margin(1e-2f));
  REQUIRE((rect.pts[1] - data.base_pts[1]).norm() == Approx(0.0f).margin(1e-2f));
  REQUIRE((rect.pts[2] - data.base_pts[2]).norm() == Approx(0.0f).margin(1e-2f));

  // Width and height constraints strictly met
  REQUIRE((rect.pts[1] - rect.pts[0]).norm() == Approx(data.target_width).margin(1e-2f));
  REQUIRE((rect.pts[2] - rect.pts[3]).norm() == Approx(data.target_width).margin(1e-2f));
  REQUIRE((rect.pts[3] - rect.pts[0]).norm() == Approx(data.target_height).margin(1e-2f));
  REQUIRE((rect.pts[2] - rect.pts[1]).norm() == Approx(data.target_height).margin(1e-2f));
}

TEST_CASE("tinyopt_optimizer_set_locked_without_trait") {
  WobblyRectangleData data;
  RectangleNoTrait<float> rect = data.CreateInitialRectangleNoTrait();
  const RectangleNoTrait<float> initial_rect = rect;
  auto loss = MakeWobblyRectangleLoss();

  Options options(Options::Solver::GradientDescent);
  options.gd.lr = 0.05f;
  options.stop.max_iters = 200;
  options.stop.min_grad_norm2 = 1e-8f;
  options.stop.min_step_norm2 = 1e-8f;

  gd::Optimizer<Vector<float, 8>> opt(options);
  // Lock the first point (coordinates 0 and 1) via lock()
  opt.lock({0, 1});
  REQUIRE(opt.hasLocked());
  REQUIRE(opt.numLocked() == 2);
  REQUIRE(opt.isLocked(0));
  REQUIRE(opt.isLocked(1));
  REQUIRE_FALSE(opt.isLocked(2));

  const auto sum = opt.Optimize(rect, loss);
  REQUIRE(sum.Succeeded());

  // First point must be strictly fixed/locked
  REQUIRE(rect.pts[0].x() == Approx(initial_rect.pts[0].x()).margin(1e-6f));
  REQUIRE(rect.pts[0].y() == Approx(initial_rect.pts[0].y()).margin(1e-6f));

  // Other points converged to nominal rectangle
  REQUIRE((rect.pts[1] - data.base_pts[1]).norm() == Approx(0.0f).margin(1e-2f));
  REQUIRE((rect.pts[2] - data.base_pts[2]).norm() == Approx(0.0f).margin(1e-2f));
  REQUIRE((rect.pts[3] - data.base_pts[3]).norm() == Approx(0.0f).margin(1e-2f));

  // Width and height constraints strictly met
  REQUIRE((rect.pts[1] - rect.pts[0]).norm() == Approx(data.target_width).margin(1e-2f));
  REQUIRE((rect.pts[2] - rect.pts[3]).norm() == Approx(data.target_width).margin(1e-2f));
  REQUIRE((rect.pts[3] - rect.pts[0]).norm() == Approx(data.target_height).margin(1e-2f));
  REQUIRE((rect.pts[2] - rect.pts[1]).norm() == Approx(data.target_height).margin(1e-2f));
}

TEST_CASE("tinyopt_optimizer_2nd_order_locked_permutation_dynamic") {
  // Test 2nd-order dynamic dimension problem with permutations and solving top-left corner
  const int n = 6;
  VecX x_init = VecX::Zero(n);
  VecX target = VecX::LinSpaced(n, 1.0, 6.0);

  auto loss = [&](const auto &x, auto &grad, auto &H) {
    const auto diff = x - target;
    if constexpr (!traits::is_nullptr_v<decltype(grad)>) {
      grad = diff;
    }
    if constexpr (!traits::is_nullptr_v<decltype(H)>) {
      H.setIdentity();
    }
    return 0.5 * diff.squaredNorm();
  };

  Options options(Options::Solver::GaussNewton);
  options.opt.min_dims_use_lock_permutation = 4;
  options.opt.min_num_locked_permutation = 1;
  options.stop.max_iters = 20;

  gn::Optimizer<MatX> opt(options);
  // Lock dimensions 1 and 4
  opt.lock({1, 4});
  REQUIRE(opt.hasLocked());
  REQUIRE(opt.numLocked() == 2);

  VecX x = x_init;
  const auto sum = opt.Optimize(x, loss);
  REQUIRE(sum.Succeeded());
  REQUIRE(opt.useLockedPermutation());

  // Locked parameters must remain exactly 0 (initial value)
  REQUIRE(x(1) == Approx(0.0).margin(1e-9));
  REQUIRE(x(4) == Approx(0.0).margin(1e-9));

  // Free parameters must converge to target
  REQUIRE(x(0) == Approx(target(0)).margin(1e-5));
  REQUIRE(x(2) == Approx(target(2)).margin(1e-5));
  REQUIRE(x(3) == Approx(target(3)).margin(1e-5));
  REQUIRE(x(5) == Approx(target(5)).margin(1e-5));
}

TEST_CASE("tinyopt_optimizer_sparse_locked_triplets") {
  // Test sparse matrix ApplyLockedStates using triplets
  const int n = 5;
  VecX x_init = VecX::Zero(n);
  VecX target = VecX::Constant(n, 2.5);

  auto loss = [&](const auto &x, auto &grad, SparseMat &H) {
    const auto diff = x - target;
    if constexpr (!traits::is_nullptr_v<decltype(grad)>) {
      grad = diff;
      SparseMat I(n, n);
      I.setIdentity();
      H = I;
    }
    return 0.5 * diff.squaredNorm();
  };

  Options options(Options::Solver::GaussNewton);
  options.linear_solver = LinearSolverMethod::LDLT;
  options.stop.max_iters = 20;

  gn::Optimizer<SparseMat> opt(options);
  // Lock index 2
  opt.lock({2});

  VecX x = x_init;
  const auto sum = opt.Optimize(x, loss);
  REQUIRE(sum.Succeeded());

  // Index 2 must be locked at 0
  REQUIRE(x(2) == Approx(0.0).margin(1e-9));

  // Other indices must converge to 2.5
  REQUIRE(x(0) == Approx(2.5).margin(1e-5));
  REQUIRE(x(1) == Approx(2.5).margin(1e-5));
  REQUIRE(x(3) == Approx(2.5).margin(1e-5));
  REQUIRE(x(4) == Approx(2.5).margin(1e-5));
}
