# Tinyopt Tutorial

This tutorial walks through the main Tinyopt workflows in a gradual way: from the shortest possible `Optimize()` call, to explicit solver configuration, to more advanced patterns such as manual accumulation, sparse optimization, multiple parameter packs, robust losses, custom parameter traits, and `ParamsWrapper` usage.

The goal is to keep the API easy to read while still exposing the mathematical machinery you need for real optimization tasks.

## Fixed-size and dynamic-size parameters

Tinyopt supports both fixed-size parameters, whose dimension is known at compile time, and dynamic-size parameters, whose dimension is determined at runtime. Fixed-size types such as `Vec2f` avoid runtime sizing and are a good fit for small, known state blocks. Dynamic types such as `VecXf` and `std::vector<float>` are useful when the number of values depends on the input data. Standard containers can also hold fixed-size blocks, for example `std::array<Vec3, 2>`.

```cpp
#include <array>
#include <vector>

#include <tinyopt/tinyopt.h>

tinyopt::Vec2f point = tinyopt::Vec2f::Zero();
std::vector<float> coefficients(8, 0.0f);
std::array<tinyopt::Vec3, 2> landmarks{};
```

For example, this short residual works with a fixed-size vector:

```cpp
tinyopt::Vec2f point = tinyopt::Vec2f::Zero();
auto residuals = [](const auto &x) { return (x - tinyopt::Vec2f(1.0f, 2.0f)).eval(); };
tinyopt::Optimize(point, residuals);
```

For dynamic parameters, initialize the desired runtime size before optimizing. Tinyopt derives the update dimension from the current value. `std::array<Vec3, 2>` is a fixed collection of two 3D blocks, with six scalar update coordinates in total.

## 1. Hello world: minimize a scalar objective

The simplest Tinyopt workflow is to write a cost or residual function and pass it directly to `Optimize()`.

```cpp
#include <tinyopt/tinyopt.h>

int main() {
  double x = 1.0;

  auto objective = [](const auto &xi) {
    return xi * xi - 2.0;
  };

  tinyopt::Optimize(x, objective);
  // x is now close to sqrt(2)
}
```

This is the shortest path for small problems. Tinyopt will infer the right differentiation path and solve the local step using the default solver settings.

## 2. Least-squares with residual vectors

For nonlinear least-squares problems, the usual pattern is to return a residual vector instead of a scalar objective. This is often more natural than manually writing the squared norm.

```cpp
#include <tinyopt/tinyopt.h>

int main() {
  tinyopt::Vec2 x = tinyopt::Vec2::Zero();
  const tinyopt::Vec2 target(3.0, -2.0);

  auto residuals = [&](const auto &value) {
    return (value - target).eval();
  };

  auto summary = tinyopt::Optimize(x, residuals);
  // x approximates the target
}
```

Here the library internally builds the residual Jacobian and solves the normal equations or a damped least-squares system. For many practical optimization tasks, this is the ideal default workflow.

## 3. Manual accumulation is faster when you know the Jacobian

The direct accumulation pattern is the fastest path when you already know the residual Jacobian and want to avoid building temporary residual vectors. Instead of returning a residual vector and letting Tinyopt differentiate it, you update the gradient and Hessian approximation directly.

```cpp
#include <tinyopt/tinyopt.h>

using namespace tinyopt;

int main() {
  Vec2 x(2.0, -1.0);
  const Vec2 target(3.0, 4.0);

  auto loss = [&](const auto &value, auto &gradient, auto &hessian) {
    const Vec2 residual = value - target;
    const Mat2 J = Mat2::Identity();

    if constexpr (!traits::is_nullptr_v<decltype(gradient)>) {
      gradient.noalias() = J.transpose() * residual;
      hessian.noalias() = J.transpose() * J;
    }

    return 0.5 * residual.squaredNorm();
  };

  auto optimizer = lm::Optimizer<Mat2>{};
  const auto summary = optimizer(x, loss);
}
```

This is the preferred pattern when the objective is structured and you want to minimize temporary allocations and avoid unnecessary Jacobian materialization.

## 4. Use an explicit optimizer object

The global helper is convenient, but sometimes you want to reuse the same solver configuration across multiple solves or inspect solver state explicitly.

```cpp
#include <tinyopt/tinyopt.h>

using namespace tinyopt;

int main() {
  Vec2 x(10.0, -10.0);
  const Vec2 target(2.5, 1.0);

  auto residuals = [&](const auto &value) {
    return (value - target).eval();
  };

  Options options;
  options.stop.max_iters = 50;
  options.log.enable = true;

  auto optimizer = lm::Optimizer<Mat2>(options);
  const auto summary = optimizer(x, residuals);
}
```

This pattern keeps the optimizer setup and the problem definition separate. It is useful when you want to:

- reuse one solver configuration across many calls,
- switch algorithms without rewriting the objective,
- monitor convergence or iterate manually.

## 5. Choosing a solver

Tinyopt exposes several optimization strategies. The correct choice depends on the structure of your objective.

### Gradient descent

```cpp
#include <tinyopt/tinyopt.h>

using namespace tinyopt;

int main() {
  Vec2 x(5.0, -2.0);
  auto loss = [&](const auto &value) {
    return (value - Vec2(1.0, 1.0)).squaredNorm();
  };

  Options options;
  options.gd.lr = 0.05;

  auto optimizer = gd::Optimizer<Vec2>(options);
  optimizer(x, loss);
}
```

### Conjugate gradient

```cpp
#include <tinyopt/tinyopt.h>

using namespace tinyopt;

int main() {
  Vec2 x(4.0, -4.0);
  auto loss = [&](const auto &value) {
    return (value - Vec2(0.0, 0.0)).squaredNorm();
  };

  Options options;
  options.cg.step_size = 0.2;

  auto optimizer = cg::Optimizer<Vec2>(options);
  optimizer(x, loss);
}
```

### BFGS and L-BFGS

```cpp
#include <tinyopt/tinyopt.h>

using namespace tinyopt;

int main() {
  Vec2 x(8.0, 8.0);
  auto loss = [&](const auto &value) {
    return (value - Vec2(2.0, 3.0)).squaredNorm();
  };

  Options options;
  options.bfgs.step_size = 0.3;
  options.lbfgs.history_size = 8;

  auto optimizer = bfgs::Optimizer<Vec2>(options);
  optimizer(x, loss);
}
```

### Gauss-Newton and Levenberg-Marquardt

These are the standard choices for least-squares problems. Gauss-Newton solves the normal equations directly, while LM adds damping to improve stability near degenerate or poorly scaled problems.

```cpp
#include <tinyopt/tinyopt.h>

using namespace tinyopt;

int main() {
  Vec2 x = Vec2::Zero();
  const Vec2 target(2.0, -1.0);

  auto residuals = [&](const auto &value) {
    return (value - target).eval();
  };

  Options options;
  options.lm.damping_init = 1e-3;

  auto gn = gn::Optimizer<Mat2>(options);
  auto lm = lm::Optimizer<Mat2>(options);

  gn(x, residuals);
  x = Vec2::Zero();
  lm(x, residuals);
}
```

For many residual-based problems, `lm::Optimizer<>` is the safest default because it remains stable when the Hessian is ill-conditioned. The dogleg variant is also available under `dl::Optimizer<>` for trust-region semantics.

To use Jacobi scaling for an ill-conditioned LM system, set `options.lm.jacobi_scaling = true`.
This applies the clamped Hessian-diagonal scaling used by Ceres; it is disabled by default.

For dense rank-deficient systems, select the opt-in `TruncatedSVD` linear solver. It computes a
pseudoinverse step by treating singular values at or below a relative cutoff as zero. Set the
cutoff on `Options`; zero selects Eigen's default. This method requires
`TINYOPT_ENABLE_LINEAR_SOLVER_SVD=ON` and is not available for sparse Hessians.

```cpp
Options options;
options.linear_solver = LinearSolverMethod::TruncatedSVD;
options.svd_relative_threshold = 1e-8;

auto optimizer = gn::Optimizer<Mat2>(options);
```

The cutoff is applied to the accumulated Hessian approximation `J.transpose() * J`, not directly to
the Jacobian.

## 6. Multi-parameter optimization

Tinyopt can optimize over multiple parameter blocks at once using `ParamsPack`.

```cpp
#include <tinyopt/tinyopt.h>

using namespace tinyopt;

int main() {
  Vec2 center(5.0, 1.0);
  Vec2 scale(2.0, 3.0);

  auto residuals = [&](const auto &c, const auto &s) {
    return (c - s).eval();
  };

  auto summary = tinyopt::Optimize(center, scale, residuals);
}
```

This pattern is useful when your model naturally decomposes into several parameter groups, such as:

- translation and rotation,
- scale and bias terms,
- pose and calibration parameters,
- separate blocks of a large state vector.

## 7. Custom parameter types

You are not limited to scalars and Eigen vectors. A custom type can use Tinyopt's default parameter trait by providing its scalar type, compile-time tangent dimension, update operator, and (for automatic differentiation) a templated `cast<T>()` member. Here is a small 2D landmark type:

```cpp
#include <tinyopt/tinyopt.h>

template <typename T>
struct Landmark2D {
  using Scalar = T;
  static constexpr tinyopt::Index Dims = 2;

  Eigen::Vector<T, 2> position = Eigen::Vector<T, 2>::Zero();

  Landmark2D &operator+=(const Eigen::Vector<T, Dims> &delta) {
    position += delta;
    return *this;
  }

  template <typename T2>
  Landmark2D<T2> cast() const {
    return {position.template cast<T2>()};
  }
};

int main() {
  Landmark2D<double> landmark;

  auto residuals = [](const auto &value) {
    return (value.position - tinyopt::Vec2(1.0, 2.0)).eval();
  };

  auto summary = tinyopt::Optimize(landmark, residuals);
}
```

The type itself defines how a local update is applied, while `cast<T>()` lets automatic differentiation evaluate the same residual with Jet scalars. If you cannot or do not want to add these members to the type, specialize `tinyopt::traits::params_trait<YourType>` instead and provide the corresponding `Scalar`, `Dims`, `PlusEq`, and cast behavior there.

## 8. `ParamsWrapper` for existing parameter storage

If the underlying parameter object already exists and you want to avoid copying it, wrap it with `ParamsWrapper`.

```cpp
#include <tinyopt/params_wrapper.h>
#include <tinyopt/tinyopt.h>

template <typename T>
struct Pose2 {
  using Scalar = T;
  using Vec2 = Eigen::Vector<T, 2>;
  using Tangent = Eigen::Vector<T, 3>;
  static constexpr tinyopt::Index Dims = 3;

  Vec2 translation = Vec2::Zero();
  Vec2 orientation = Vec2(1, 0);

  Pose2 &operator+=(const Tangent &delta) {
    translation += delta.template head<2>();
    const T tangent = delta[2];
    const T denominator = T(1) + tangent * tangent;
    const T cosine = (T(1) - tangent * tangent) / denominator;
    const T sine = T(2) * tangent / denominator;
    const Vec2 previous = orientation;
    orientation[0] = cosine * previous[0] - sine * previous[1];
    orientation[1] = sine * previous[0] + cosine * previous[1];
    return *this;
  }

  template <typename T2>
  Pose2<T2> cast() const {
    return {translation.template cast<T2>(), orientation.template cast<T2>()};
  }
};

int main() {
  Pose2<double> pose;
  tinyopt::ParamsWrapper<Pose2<double> &> parameters(pose);

  auto residuals = [&](const auto &p) {
    Eigen::Vector<typename std::decay_t<decltype(p.params)>::Scalar, 4> r;
    r.template head<2>() = p.params.translation - tinyopt::Vec2(2.0, -1.0);
    r.template tail<2>() = p.params.orientation - tinyopt::Vec2(0.8, 0.6);
    return r;
  };

  auto summary = tinyopt::Optimize(parameters, residuals);
}
```

`ParamsWrapper<MyParameters>` owns the value, while `ParamsWrapper<MyParameters &>` keeps the original object by reference. This is especially useful when the parameter block already lives in user-managed storage and you do not want an extra copy.

## 9. Robust losses for outliers

The plain quadratic L2 loss can be too sensitive to a few large residuals. Tinyopt includes robust losses such as Huber and Cauchy, which adapt the weighting of residuals.

```cpp
#include <tinyopt/tinyopt.h>
#include <tinyopt/losses/robust_norms.h>

using namespace tinyopt;

int main() {
  Vec2 x = Vec2::Zero();
  const Vec2 target(1.0, 1.0);

  auto residuals = [&](const auto &value) {
    const auto r = (value - target).eval();
    return losses::Huber(r, 0.7);
  };

  auto summary = tinyopt::Optimize(x, residuals);
}
```

Robust losses are especially useful in the presence of outliers or data that is not perfectly Gaussian.

## 10. Sparse optimization

For large systems where the Hessian matrix is mostly zero, allocating and factoring a dense matrix becomes prohibitive. Tinyopt allows you to optimize sparse systems by accepting a `SparseMat &hessian` (or `SparseMatrix<T> &hessian`) in your accumulation function.

When `Optimize()` detects a `SparseMat &` parameter in the cost function signature, it automatically dispatches to a sparse linear solver (such as Eigen's `SimplicialLDLT` or SuiteSparse when enabled) without requiring manual solver instantiation:

```cpp
#include <vector>
#include <tinyopt/tinyopt.h>

using namespace tinyopt;

int main() {
  VecX x = VecX::Constant(5, 1.0);
  const VecX target = VecX::Constant(5, 3.0);

  auto loss = [&](auto &x, auto &grad, SparseMat &hessian) {
    const VecX res = x - target;

    // Populate gradient and sparse Hessian when requested
    if constexpr (!traits::is_nullptr_v<decltype(grad)>) {
      grad = res;

      // Populate sparse Hessian triplets
      std::vector<Eigen::Triplet<double>> triplets;
      triplets.reserve(x.size());
      for (int i = 0; i < x.size(); ++i) {
        triplets.emplace_back(i, i, 1.0);
      }
      hessian.setFromTriplets(triplets.begin(), triplets.end());
    }

    return 0.5 * res.squaredNorm();
  };

  auto summary = tinyopt::Optimize(x, loss);
}
```

You can also assemble `hessian` from a sparse Jacobian (e.g., `hessian = Js.transpose() * Js`) or update non-zero entries directly with `hessian.coeffRef(row, col)`.

## 11. Recommended workflow

A good way to work with Tinyopt is:

1. Start with a scalar or residual-based objective and call `tinyopt::Optimize()`.
2. If the problem is structured and you know the Jacobian, move to manual accumulation (or use `SparseMat &hessian` for sparse problems).
3. Reuse a solver configuration via `gn::Optimizer<>`, `lm::Optimizer<>`, `gd::Optimizer<>`, `cg::Optimizer<>`, `bfgs::Optimizer<>`, or `dl::Optimizer<>` when the optimization is repeated or the solver must be controlled explicitly.
4. Use `ParamsPack` for multiple parameters, `ParamsWrapper` for in-place parameter state, and custom `params_trait` definitions for custom objects or manifold-type states.
5. Add robust weighting when outliers are expected, and use covariance estimates when uncertainty quantification matters.

This keeps the library ergonomic for quick experiments while preserving the direct, near-zero-allocation optimization patterns needed for production numerical work.
