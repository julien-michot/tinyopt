// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>
#include <vector>

#include <ceres/ceres.h>
#include <ceres/gradient_problem.h>
#include <ceres/gradient_problem_solver.h>
#include <Eigen/Core>

#ifdef CHECK
#undef CHECK
#endif

#include <tinyopt/tinyopt.h>

namespace tinyopt::ceres_exact {

// =============================================================================
// Problem 1: Nonlinear Coupled Polynomial System (4D)
// -----------------------------------------------------------------------------
// r_0(x) = x_0^2 + x_1 - 2
// r_1(x) = x_0 + x_1^2 + x_2 - 3
// r_2(x) = x_1 + x_2^2 + x_3 - 3
// r_3(x) = x_2 + x_3^2 - 2
// Minimum: x* = [1, 1, 1, 1]^T, r(x*) = 0.
// Initial guess: x_0 = [2, 2, 2, 2]^T.
// Iterations: ~4-8 iterations across solvers.
// =============================================================================
struct NonlinearSystem4D {
  static constexpr int Dims = 4;
  using Vec = Eigen::Matrix<double, Dims, 1>;
  using Mat = Eigen::Matrix<double, Dims, Dims>;

  static Vec Initial() { return Vec::Constant(1.5); }
  static Vec Target() { return Vec::Constant(1.0); }

  // Tinyopt direct accumulation / residuals
  template <typename Derived>
  static auto Residuals(const Eigen::MatrixBase<Derived>& x) {
    using Scalar = typename Derived::Scalar;
    Eigen::Matrix<Scalar, Dims, 1> r;
    r[0] = x[0] * x[0] + x[1] - Scalar(2.0);
    r[1] = x[0] + x[1] * x[1] + x[2] - Scalar(3.0);
    r[2] = x[1] + x[2] * x[2] + x[3] - Scalar(3.0);
    r[3] = x[2] + x[3] * x[3] - Scalar(2.0);
    return r;
  }

  template <typename Parameters, typename Gradient, typename Hessian>
  static Cost Accumulate(const Parameters& x, Gradient& grad, Hessian& hess) {
    using Scalar = typename Parameters::Scalar;
    const auto r = Residuals(x);
    if constexpr (!traits::is_nullptr_v<decltype(grad)>) {
      Eigen::Matrix<Scalar, Dims, Dims> J;
      J.setZero();
      J(0, 0) = Scalar(2.0) * x[0];
      J(0, 1) = Scalar(1.0);

      J(1, 0) = Scalar(1.0);
      J(1, 1) = Scalar(2.0) * x[1];
      J(1, 2) = Scalar(1.0);

      J(2, 1) = Scalar(1.0);
      J(2, 2) = Scalar(2.0) * x[2];
      J(2, 3) = Scalar(1.0);

      J(3, 2) = Scalar(1.0);
      J(3, 3) = Scalar(2.0) * x[3];

      grad.noalias() = J.transpose() * r;
      if constexpr (!traits::is_nullptr_v<decltype(hess)>) {
        hess.noalias() = J.transpose() * J;
      }
    }
    return Cost(Scalar(0.5) * r.squaredNorm(), Dims);
  }

  template <typename Parameters, typename Gradient>
  static typename Parameters::Scalar EvaluateObjective(const Parameters& x, Gradient& grad) {
    using Scalar = typename Parameters::Scalar;
    const auto r = Residuals(x);
    if constexpr (!traits::is_nullptr_v<decltype(grad)>) {
      Eigen::Matrix<Scalar, Dims, Dims> J;
      J.setZero();
      J(0, 0) = Scalar(2.0) * x[0];
      J(0, 1) = Scalar(1.0);

      J(1, 0) = Scalar(1.0);
      J(1, 1) = Scalar(2.0) * x[1];
      J(1, 2) = Scalar(1.0);

      J(2, 1) = Scalar(1.0);
      J(2, 2) = Scalar(2.0) * x[2];
      J(2, 3) = Scalar(1.0);

      J(3, 2) = Scalar(1.0);
      J(3, 3) = Scalar(2.0) * x[3];

      grad.noalias() = J.transpose() * r;
    }
    return Scalar(0.5) * r.squaredNorm();
  }

  // Ceres CostFunction for NLLS (LM, DogLeg)
  class CeresCostFunction final : public ceres::SizedCostFunction<Dims, Dims> {
   public:
    bool Evaluate(double const* const* parameters, double* residuals,
                  double** jacobians) const override {
      const Eigen::Map<const Vec> x(parameters[0]);
      Eigen::Map<Vec> r(residuals);
      r = NonlinearSystem4D::Residuals(x);
      if (jacobians != nullptr && jacobians[0] != nullptr) {
        Eigen::Map<Eigen::Matrix<double, Dims, Dims, Eigen::RowMajor>> J(jacobians[0]);
        J.setZero();
        J(0, 0) = 2.0 * x[0];
        J(0, 1) = 1.0;
        J(1, 0) = 1.0;
        J(1, 1) = 2.0 * x[1];
        J(1, 2) = 1.0;
        J(2, 1) = 1.0;
        J(2, 2) = 2.0 * x[2];
        J(2, 3) = 1.0;
        J(3, 2) = 1.0;
        J(3, 3) = 2.0 * x[3];
      }
      return true;
    }
  };

  // Ceres FirstOrderFunction for Line Search (BFGS, LBFGS, CG, GD)
  class CeresFirstOrder final : public ceres::FirstOrderFunction {
   public:
    int NumParameters() const override { return Dims; }
    bool Evaluate(const double* const parameters, double* cost, double* gradient) const override {
      const Eigen::Map<const Vec> x(parameters);
      const Vec r = NonlinearSystem4D::Residuals(x);
      if (cost != nullptr) *cost = 0.5 * r.squaredNorm();
      if (gradient != nullptr) {
        Eigen::Matrix<double, Dims, Dims> J;
        J.setZero();
        J(0, 0) = 2.0 * x[0];
        J(0, 1) = 1.0;
        J(1, 0) = 1.0;
        J(1, 1) = 2.0 * x[1];
        J(1, 2) = 1.0;
        J(2, 1) = 1.0;
        J(2, 2) = 2.0 * x[2];
        J(2, 3) = 1.0;
        J(3, 2) = 1.0;
        J(3, 3) = 2.0 * x[3];
        Eigen::Map<Vec> g(gradient);
        g.noalias() = J.transpose() * r;
      }
      return true;
    }
  };
};

// =============================================================================
// Problem 2: Extended Powell Singular Function (4D)
// -----------------------------------------------------------------------------
// r_0(x) = x_0 + 10 x_1
// r_1(x) = sqrt(5) * (x_2 - x_3)
// r_2(x) = (x_1 - 2 x_2)^2
// r_3(x) = sqrt(10) * (x_0 - x_3)^2
// Minimum: x* = [0, 0, 0, 0]^T, r(x*) = 0.
// Initial guess: x_0 = [3, -1, 0, 1]^T.
// Iterations: ~8-16 iterations. Rank-deficient Jacobian at solution.
// =============================================================================
struct Powell4D {
  static constexpr int Dims = 4;
  using Vec = Eigen::Matrix<double, Dims, 1>;
  using Mat = Eigen::Matrix<double, Dims, Dims>;

  static Vec Initial() {
    Vec x;
    x << 3.0, -1.0, 0.0, 1.0;
    return x;
  }
  static Vec Target() { return Vec::Zero(); }

  template <typename Derived>
  static auto Residuals(const Eigen::MatrixBase<Derived>& x) {
    using Scalar = typename Derived::Scalar;
    Eigen::Matrix<Scalar, Dims, 1> r;
    r[0] = x[0] + Scalar(10.0) * x[1];
    r[1] = std::sqrt(Scalar(5.0)) * (x[2] - x[3]);
    const Scalar d1 = x[1] - Scalar(2.0) * x[2];
    r[2] = d1 * d1;
    const Scalar d2 = x[0] - x[3];
    r[3] = std::sqrt(Scalar(10.0)) * d2 * d2;
    return r;
  }

  template <typename Parameters, typename Gradient, typename Hessian>
  static Cost Accumulate(const Parameters& x, Gradient& grad, Hessian& hess) {
    using Scalar = typename Parameters::Scalar;
    const auto r = Residuals(x);
    if constexpr (!traits::is_nullptr_v<decltype(grad)>) {
      Eigen::Matrix<Scalar, Dims, Dims> J;
      J.setZero();
      J(0, 0) = Scalar(1.0);
      J(0, 1) = Scalar(10.0);

      J(1, 2) = std::sqrt(Scalar(5.0));
      J(1, 3) = -std::sqrt(Scalar(5.0));

      const Scalar d1 = x[1] - Scalar(2.0) * x[2];
      J(2, 1) = Scalar(2.0) * d1;
      J(2, 2) = -Scalar(4.0) * d1;

      const Scalar d2 = x[0] - x[3];
      J(3, 0) = Scalar(2.0) * std::sqrt(Scalar(10.0)) * d2;
      J(3, 3) = -Scalar(2.0) * std::sqrt(Scalar(10.0)) * d2;

      grad.noalias() = J.transpose() * r;
      if constexpr (!traits::is_nullptr_v<decltype(hess)>) {
        hess.noalias() = J.transpose() * J;
      }
    }
    return Cost(Scalar(0.5) * r.squaredNorm(), Dims);
  }

  template <typename Parameters, typename Gradient>
  static typename Parameters::Scalar EvaluateObjective(const Parameters& x, Gradient& grad) {
    using Scalar = typename Parameters::Scalar;
    const auto r = Residuals(x);
    if constexpr (!traits::is_nullptr_v<decltype(grad)>) {
      Eigen::Matrix<Scalar, Dims, Dims> J;
      J.setZero();
      J(0, 0) = Scalar(1.0);
      J(0, 1) = Scalar(10.0);

      J(1, 2) = std::sqrt(Scalar(5.0));
      J(1, 3) = -std::sqrt(Scalar(5.0));

      const Scalar d1 = x[1] - Scalar(2.0) * x[2];
      J(2, 1) = Scalar(2.0) * d1;
      J(2, 2) = -Scalar(4.0) * d1;

      const Scalar d2 = x[0] - x[3];
      J(3, 0) = Scalar(2.0) * std::sqrt(Scalar(10.0)) * d2;
      J(3, 3) = -Scalar(2.0) * std::sqrt(Scalar(10.0)) * d2;

      grad.noalias() = J.transpose() * r;
    }
    return Scalar(0.5) * r.squaredNorm();
  }

  class CeresCostFunction final : public ceres::SizedCostFunction<Dims, Dims> {
   public:
    bool Evaluate(double const* const* parameters, double* residuals,
                  double** jacobians) const override {
      const Eigen::Map<const Vec> x(parameters[0]);
      Eigen::Map<Vec> r(residuals);
      r = Powell4D::Residuals(x);
      if (jacobians != nullptr && jacobians[0] != nullptr) {
        Eigen::Map<Eigen::Matrix<double, Dims, Dims, Eigen::RowMajor>> J(jacobians[0]);
        J.setZero();
        J(0, 0) = 1.0;
        J(0, 1) = 10.0;
        J(1, 2) = std::sqrt(5.0);
        J(1, 3) = -std::sqrt(5.0);
        const double d1 = x[1] - 2.0 * x[2];
        J(2, 1) = 2.0 * d1;
        J(2, 2) = -4.0 * d1;
        const double d2 = x[0] - x[3];
        J(3, 0) = 2.0 * std::sqrt(10.0) * d2;
        J(3, 3) = -2.0 * std::sqrt(10.0) * d2;
      }
      return true;
    }
  };

  class CeresFirstOrder final : public ceres::FirstOrderFunction {
   public:
    int NumParameters() const override { return Dims; }
    bool Evaluate(const double* const parameters, double* cost, double* gradient) const override {
      const Eigen::Map<const Vec> x(parameters);
      const Vec r = Powell4D::Residuals(x);
      if (cost != nullptr) *cost = 0.5 * r.squaredNorm();
      if (gradient != nullptr) {
        Eigen::Matrix<double, Dims, Dims> J;
        J.setZero();
        J(0, 0) = 1.0;
        J(0, 1) = 10.0;
        J(1, 2) = std::sqrt(5.0);
        J(1, 3) = -std::sqrt(5.0);
        const double d1 = x[1] - 2.0 * x[2];
        J(2, 1) = 2.0 * d1;
        J(2, 2) = -4.0 * d1;
        const double d2 = x[0] - x[3];
        J(3, 0) = 2.0 * std::sqrt(10.0) * d2;
        J(3, 3) = -2.0 * std::sqrt(10.0) * d2;
        Eigen::Map<Vec> g(gradient);
        g.noalias() = J.transpose() * r;
      }
      return true;
    }
  };
};

// =============================================================================
// Problem 3: Coupled Mildly-Nonlinear Quartic (6D)
// -----------------------------------------------------------------------------
// Well-conditioned nonlinear problem suitable for Gradient Descent, CG, BFGS,
// LBFGS, LM, DogLeg, GN.
// r_i(x) = sqrt(c_i) * (x_i - target_i)       for i = 0..5
// r_{6+i}(x) = 0.5 * (x_{i+1} - x_i^2)         for i = 0..4
// Total 11 residuals, 6 parameters.
// Minimum near target: converges in 4-15 iterations for all solvers.
// =============================================================================
struct Quartic6D {
  static constexpr int Dims = 6;
  static constexpr int NumRes = 11;
  using Vec = Eigen::Matrix<double, Dims, 1>;
  using ResVec = Eigen::Matrix<double, NumRes, 1>;

  static Vec Initial() { return Vec::Zero(); }
  static Vec Target() {
    Vec t;
    t << 1.0, 1.0, 1.0, 1.0, 1.0, 1.0;
    return t;
  }

  template <typename Derived>
  static auto Residuals(const Eigen::MatrixBase<Derived>& x) {
    using Scalar = typename Derived::Scalar;
    Eigen::Matrix<Scalar, NumRes, 1> r;
    // Linear component
    for (int i = 0; i < Dims; ++i) {
      const Scalar c = Scalar(1.0) + Scalar(0.5) * Scalar(i);
      r[i] = std::sqrt(c) * (x[i] - Scalar(1.0));
    }
    // Nonlinear coupling
    for (int i = 0; i < Dims - 1; ++i) {
      r[Dims + i] = Scalar(0.5) * (x[i + 1] - x[i] * x[i]);
    }
    return r;
  }

  template <typename Parameters, typename Gradient, typename Hessian>
  static Cost Accumulate(const Parameters& x, Gradient& grad, Hessian& hess) {
    using Scalar = typename Parameters::Scalar;
    const auto r = Residuals(x);
    if constexpr (!traits::is_nullptr_v<decltype(grad)>) {
      Eigen::Matrix<Scalar, NumRes, Dims> J;
      J.setZero();
      for (int i = 0; i < Dims; ++i) {
        const Scalar c = Scalar(1.0) + Scalar(0.5) * Scalar(i);
        J(i, i) = std::sqrt(c);
      }
      for (int i = 0; i < Dims - 1; ++i) {
        J(Dims + i, i) = -x[i];
        J(Dims + i, i + 1) = Scalar(0.5);
      }
      grad.noalias() = J.transpose() * r;
      if constexpr (!traits::is_nullptr_v<decltype(hess)>) {
        hess.noalias() = J.transpose() * J;
      }
    }
    return Cost(Scalar(0.5) * r.squaredNorm(), NumRes);
  }

  template <typename Parameters, typename Gradient>
  static typename Parameters::Scalar EvaluateObjective(const Parameters& x, Gradient& grad) {
    using Scalar = typename Parameters::Scalar;
    const auto r = Residuals(x);
    if constexpr (!traits::is_nullptr_v<decltype(grad)>) {
      Eigen::Matrix<Scalar, NumRes, Dims> J;
      J.setZero();
      for (int i = 0; i < Dims; ++i) {
        const Scalar c = Scalar(1.0) + Scalar(0.5) * Scalar(i);
        J(i, i) = std::sqrt(c);
      }
      for (int i = 0; i < Dims - 1; ++i) {
        J(Dims + i, i) = -x[i];
        J(Dims + i, i + 1) = Scalar(0.5);
      }
      grad.noalias() = J.transpose() * r;
    }
    return Scalar(0.5) * r.squaredNorm();
  }

  class CeresCostFunction final : public ceres::SizedCostFunction<NumRes, Dims> {
   public:
    bool Evaluate(double const* const* parameters, double* residuals,
                  double** jacobians) const override {
      const Eigen::Map<const Vec> x(parameters[0]);
      Eigen::Map<ResVec> r(residuals);
      r = Quartic6D::Residuals(x);
      if (jacobians != nullptr && jacobians[0] != nullptr) {
        Eigen::Map<Eigen::Matrix<double, NumRes, Dims, Eigen::RowMajor>> J(jacobians[0]);
        J.setZero();
        for (int i = 0; i < Dims; ++i) {
          const double c = 1.0 + 0.5 * static_cast<double>(i);
          J(i, i) = std::sqrt(c);
        }
        for (int i = 0; i < Dims - 1; ++i) {
          J(Dims + i, i) = -x[i];
          J(Dims + i, i + 1) = 0.5;
        }
      }
      return true;
    }
  };

  class CeresFirstOrder final : public ceres::FirstOrderFunction {
   public:
    int NumParameters() const override { return Dims; }
    bool Evaluate(const double* const parameters, double* cost, double* gradient) const override {
      const Eigen::Map<const Vec> x(parameters);
      const ResVec r = Quartic6D::Residuals(x);
      if (cost != nullptr) *cost = 0.5 * r.squaredNorm();
      if (gradient != nullptr) {
        Eigen::Matrix<double, NumRes, Dims> J;
        J.setZero();
        for (int i = 0; i < Dims; ++i) {
          const double c = 1.0 + 0.5 * static_cast<double>(i);
          J(i, i) = std::sqrt(c);
        }
        for (int i = 0; i < Dims - 1; ++i) {
          J(Dims + i, i) = -x[i];
          J(Dims + i, i + 1) = 0.5;
        }
        Eigen::Map<Vec> g(gradient);
        g.noalias() = J.transpose() * r;
      }
      return true;
    }
  };
};

// =============================================================================
// Problem 4: Strictly Convex Quadratic (4D)
// -----------------------------------------------------------------------------
// f(x) = 0.5 * (x - t)^T A (x - t)
// Minimum at target: t = [1, 2, 3, 4]^T.
// Initial guess: x_0 = [0, 0, 0, 0]^T.
// Well-conditioned, takes 4-10 iterations.
// =============================================================================
struct Quadratic4D {
  static constexpr int Dims = 4;
  using Vec = Eigen::Matrix<double, Dims, 1>;
  using Mat = Eigen::Matrix<double, Dims, Dims>;

  static Vec Initial() { return Vec::Zero(); }
  static Vec Target() {
    Vec t;
    t << 1.0, 2.0, 3.0, 4.0;
    return t;
  }

  static Mat MatrixA() {
    Mat A;
    A << 2.0, 0.5, 0.0, 0.0, 0.5, 2.0, 0.5, 0.0, 0.0, 0.5, 2.0, 0.5, 0.0, 0.0, 0.5, 2.0;
    return A;
  }

  template <typename Parameters, typename Gradient>
  static typename Parameters::Scalar EvaluateObjective(const Parameters& x, Gradient& grad) {
    using Scalar = typename Parameters::Scalar;
    const Vec diff = x - Target();
    const Mat A = MatrixA();
    if constexpr (!traits::is_nullptr_v<decltype(grad)>) {
      grad.noalias() = (A * diff).template cast<Scalar>();
    }
    return Scalar(0.5) * diff.dot(A * diff);
  }

  class CeresFirstOrder final : public ceres::FirstOrderFunction {
   public:
    int NumParameters() const override { return Dims; }
    bool Evaluate(const double* const parameters, double* cost, double* gradient) const override {
      const Eigen::Map<const Vec> x(parameters);
      const Vec diff = x - Quadratic4D::Target();
      const Mat A = Quadratic4D::MatrixA();
      if (cost != nullptr) *cost = 0.5 * diff.dot(A * diff);
      if (gradient != nullptr) {
        Eigen::Map<Vec> g(gradient);
        g.noalias() = A * diff;
      }
      return true;
    }
  };
};

}  // namespace tinyopt::ceres_exact
