// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <vector>

#if CATCH2_VERSION == 2
#include <catch2/catch.hpp>
#else
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#endif

#include <tinyopt/tinyopt.h>

using Catch::Approx;
using namespace tinyopt;

namespace {

// Parameter struct defining the reduced() trait returning 1 pair with first == 0
template <typename T = double>
struct CameraPointProblem {
  using Scalar = T;
  static constexpr Index Dims = 5;  // 2 camera parameters (head), 3 landmark parameters (tail)

  Vector<T, 5> values;

  CameraPointProblem() { values.setZero(); }
  explicit CameraPointProblem(const Vector<T, 5> &v) : values(v) {}

  // reduced() returning 1 pair with first == 0: {0, 2}
  std::pair<Index, Index> reduced() const { return {0, 2}; }

  template <typename T2>
  CameraPointProblem<T2> cast() const {
    return CameraPointProblem<T2>(values.template cast<T2>());
  }

  template <typename DeltaType>
  CameraPointProblem &operator+=(const DeltaType &delta) {
    values += delta;
    return *this;
  }
};

// Parameter struct without the reduced() trait
template <typename T = double>
struct CameraPointProblemNoTrait {
  using Scalar = T;
  static constexpr Index Dims = 5;

  Vector<T, 5> values;

  CameraPointProblemNoTrait() { values.setZero(); }
  explicit CameraPointProblemNoTrait(const Vector<T, 5> &v) : values(v) {}

  template <typename T2>
  CameraPointProblemNoTrait<T2> cast() const {
    return CameraPointProblemNoTrait<T2>(values.template cast<T2>());
  }

  template <typename DeltaType>
  CameraPointProblemNoTrait &operator+=(const DeltaType &delta) {
    values += delta;
    return *this;
  }
};

}  // namespace

TEST_CASE("schur_reduced_trait", "[schur][traits]") {
  SECTION("Default reduced trait is false for basic types") {
    REQUIRE_FALSE(traits::has_reduced_v<float>);
    REQUIRE_FALSE(traits::has_reduced_v<double>);
    REQUIRE_FALSE(traits::has_reduced_v<Vec3>);
    REQUIRE_FALSE(traits::has_reduced_v<VecX>);
    REQUIRE_FALSE(traits::has_reduced_v<std::vector<double>>);
    REQUIRE_FALSE(traits::has_reduced_v<CameraPointProblemNoTrait<double>>);
  }

  SECTION("User defined reduced trait is recognized") {
    REQUIRE(traits::has_reduced_v<CameraPointProblem<double>>);
    CameraPointProblem<double> p;
    auto red = traits::reduced(p);
    REQUIRE(red.first == 0);
    REQUIRE(red.second == 2);
  }
}

TEST_CASE("schur_linear_solver_unit", "[schur][linear]") {
  // Construct a sparse 5x5 system: 2 reduced (head), 3 eliminated (tail)
  const Index N = 2;
  const Index D = 5;

  std::vector<Eigen::Triplet<double>> triplets;

  // Block B (2x2)
  triplets.emplace_back(0, 0, 4.0);
  triplets.emplace_back(0, 1, 1.0);
  triplets.emplace_back(1, 0, 1.0);
  triplets.emplace_back(1, 1, 3.0);

  // Coupling Block E (2x3) and E^T (3x2)
  triplets.emplace_back(0, 2, 0.5);
  triplets.emplace_back(2, 0, 0.5);
  triplets.emplace_back(0, 3, 0.2);
  triplets.emplace_back(3, 0, 0.2);
  triplets.emplace_back(1, 3, 0.3);
  triplets.emplace_back(3, 1, 0.3);
  triplets.emplace_back(1, 4, 0.4);
  triplets.emplace_back(4, 1, 0.4);

  // Block C (3x3)
  triplets.emplace_back(2, 2, 5.0);
  triplets.emplace_back(2, 3, 0.5);
  triplets.emplace_back(3, 2, 0.5);
  triplets.emplace_back(3, 3, 6.0);
  triplets.emplace_back(4, 4, 4.0);

  SparseMat H(D, D);
  H.setFromTriplets(triplets.begin(), triplets.end());

  Vector<double, D> x_expected;
  x_expected << 1.0, -2.0, 3.0, 0.5, -1.5;
  Vector<double, D> b = H * x_expected;

  Options options(Options::Solver::GaussNewton);
  options.linear.method = LinearSolverMethod::LDLT;
  gn::Optimizer<SparseMat> opt(options);

  opt.InitWith(Vector<double, D>::Zero(), H);
  opt.setReduced(N);

  REQUIRE(opt.hasReduced());
  REQUIRE(opt.numReduced() == N);

  SECTION("SparseLinearSolvers computes exact Schur solution") {
    SparseLinearSolvers<SparseMat> schur;
    REQUIRE(schur.Analyze(H, N));
    auto maybe_step = schur.Solve(H, -b);
    REQUIRE(maybe_step.has_value());
    REQUIRE((*maybe_step + x_expected).norm() < 1e-10);

    // With explicit solver
    SparseLinearSolvers<SparseMat> ws;
    REQUIRE(ws.Analyze(H, N));
    auto maybe_step_ws = ws.Solve(H, -b);
    REQUIRE(maybe_step_ws.has_value());
    REQUIRE((*maybe_step_ws + x_expected).norm() < 1e-10);

    LinearSolverOptions no_schur_options{.allow_schur = false};
    SparseLinearSolvers<SparseMat> no_schur(no_schur_options);
    REQUIRE_FALSE(no_schur.Analyze(H, N));
    REQUIRE_FALSE(no_schur.Solve(H, -b).has_value());
    REQUIRE(no_schur.Analyze(H));
    REQUIRE(no_schur.Solve(H, -b).has_value());

#if defined(TINYOPT_ENABLE_LINEAR_SOLVER_INCOMPLETE_CHOLESKY)
    // With IncompleteCholesky solver
    LinearSolverOptions opt_ic{.method = LinearSolverMethod::IncompleteCholesky,
                               .initial_shift = 0.0};
    SparseLinearSolvers<SparseMat> ws_ic(opt_ic);
    REQUIRE(ws_ic.Analyze(H, N));
    auto maybe_step_ic = ws_ic.Solve(H, -b);
    REQUIRE(maybe_step_ic.has_value());
    REQUIRE((*maybe_step_ic + x_expected).norm() < 0.1);
#endif
  }

  SECTION("SolveLinear member computes exact solution") {
    auto maybe_step = opt.SolveLinear(b);
    REQUIRE(maybe_step.has_value());
    REQUIRE((*maybe_step + x_expected).norm() < 1e-10);
  }

  SECTION("SolveLinear dispatches to Schur solve when reduced is active") {
    auto maybe_step = opt.SolveLinear(b);
    REQUIRE(maybe_step.has_value());
    REQUIRE((*maybe_step + x_expected).norm() < 1e-10);
  }

  SECTION("Singular block C returns nullopt") {
    SparseMat H_singular = H;
    // Make row/col 4 zero
    std::vector<Eigen::Triplet<double>> sing_triplets;
    for (int k = 0; k < H.outerSize(); ++k) {
      for (SparseMat::InnerIterator it(H, k); it; ++it) {
        if (it.row() != 4 && it.col() != 4) {
          sing_triplets.emplace_back(it.row(), it.col(), it.value());
        }
      }
    }
    H_singular.setFromTriplets(sing_triplets.begin(), sing_triplets.end());
    opt.InitWith(Vector<double, D>::Zero(), H_singular);
    auto maybe_step = opt.SolveLinear(b);
    REQUIRE_FALSE(maybe_step.has_value());
  }

  SECTION("Non-finite rhs returns nullopt") {
    Vector<double, D> b_nan = b;
    b_nan(0) = std::numeric_limits<double>::quiet_NaN();
    auto maybe_step = opt.SolveLinear(b_nan);
    REQUIRE_FALSE(maybe_step.has_value());
  }

  SECTION("SolveLinear rejects locked permutation with Schur solve") {
    Options opt_perm_options(Options::Solver::GaussNewton);
    opt_perm_options.opt.min_locked_ratio = 0.01f;
    gn::Optimizer<MatX> opt_perm(opt_perm_options);
    const MatX H_dense = H.toDense();
    opt_perm.InitWith(Vector<double, D>::Zero(), H_dense);
    opt_perm.lock({3});
    opt_perm.InitLockedPermutation(D);
    REQUIRE(opt_perm.useLockedPermutation());
    opt_perm.setReduced(N);
    REQUIRE_THROWS_AS(opt_perm.SolveLinear(b), std::runtime_error);
  }
}

TEST_CASE("dense Schur solver computes exact solution", "[schur][dense]") {
  MatX H(4, 4);
  H << 8.0, 1.0, 0.5, 0.0, 1.0, 7.0, 0.0, 0.2, 0.5, 0.0, 5.0, 1.0, 0.0, 0.2, 1.0, 6.0;
  VecX expected(4);
  expected << 1.0, -2.0, 0.5, 3.0;
  const VecX rhs = H * expected;

  DenseLinearSolvers<MatX> solver;
  REQUIRE(solver.Analyze(H, 2));
  const auto solution = solver.Solve(H, rhs);
  REQUIRE(solution.has_value());
  REQUIRE((*solution - expected).norm() < 1e-10);

  REQUIRE(solver.Analyze(H));
  const auto full_solution = solver.Solve(H, rhs);
  REQUIRE(full_solution.has_value());
  REQUIRE((*full_solution - expected).norm() < 1e-10);
  REQUIRE_FALSE(solver.Analyze(H, -1));
  REQUIRE_FALSE(solver.Solve(H, rhs).has_value());
  REQUIRE(solver.Analyze(H, H.rows() - 1));
  REQUIRE_FALSE(solver.Analyze(H, H.rows()));
  REQUIRE_FALSE(solver.Solve(H, rhs).has_value());

  MatX singular = H;
  singular.bottomRightCorner(2, 2).setZero();
  REQUIRE(solver.Analyze(singular, 2));
  REQUIRE_FALSE(solver.Solve(singular, rhs).has_value());

  LinearSolverOptions no_schur_options{.allow_schur = false};
  DenseLinearSolvers<MatX> no_schur(no_schur_options);
  REQUIRE_FALSE(no_schur.Analyze(H, 2));
  REQUIRE_FALSE(no_schur.Solve(H, rhs).has_value());
  REQUIRE(no_schur.Analyze(H));
  REQUIRE(no_schur.Solve(H, rhs).has_value());
}

TEST_CASE("dense and sparse Schur solvers resize between partitions", "[schur][resize]") {
  MatX H4 = MatX::Identity(4, 4) * 5.0;
  VecX x4 = VecX::LinSpaced(4, 1.0, 4.0);
  VecX rhs4 = H4 * x4;

  MatX H3 = MatX::Identity(3, 3) * 7.0;
  VecX x3 = VecX::LinSpaced(3, -1.0, 1.0);
  VecX rhs3 = H3 * x3;

  DenseLinearSolvers<MatX> dense_solver;
  REQUIRE(dense_solver.Analyze(H4, 2));
  auto dense_solution = dense_solver.Solve(H4, rhs4);
  REQUIRE(dense_solution.has_value());
  REQUIRE((*dense_solution - x4).norm() < 1e-10);
  REQUIRE(dense_solver.Analyze(H3, 1));
  dense_solution = dense_solver.Solve(H3, rhs3);
  REQUIRE(dense_solution.has_value());
  REQUIRE((*dense_solution - x3).norm() < 1e-10);

  SparseLinearSolvers<SparseMat> sparse_solver;
  SparseMat sparse_H4 = H4.sparseView();
  REQUIRE(sparse_solver.Analyze(sparse_H4, 2));
  auto sparse_solution = sparse_solver.Solve(sparse_H4, rhs4);
  REQUIRE(sparse_solution.has_value());
  REQUIRE((*sparse_solution - x4).norm() < 1e-10);

  // Inverting the full matrix must invert the whole matrix, not the Schur-eliminated block,
  // and must leave the Schur analysis usable for subsequent solves.
  const auto sparse_inverse = sparse_solver.Inverse(sparse_H4);
  REQUIRE(sparse_inverse.has_value());
  REQUIRE((sparse_inverse->toDense() - H4.inverse()).norm() < 1e-10);
  sparse_solution = sparse_solver.Solve(sparse_H4, rhs4);
  REQUIRE(sparse_solution.has_value());
  REQUIRE((*sparse_solution - x4).norm() < 1e-10);

  SparseMat sparse_H3 = H3.sparseView();
  REQUIRE(sparse_solver.Analyze(sparse_H3, 1));
  sparse_solution = sparse_solver.Solve(sparse_H3, rhs3);
  REQUIRE(sparse_solution.has_value());
  REQUIRE((*sparse_solution - x3).norm() < 1e-10);
}

TEST_CASE("schur_optimization_with_reduced_trait", "[schur][optimizer]") {
  const Vector<double, 5> target{1.5, -2.0, 3.0, 0.5, 4.0};

  auto loss = [&](const auto &param, auto &grad, SparseMat &H) {
    const auto &x = param.values;
    const auto diff = x - target;

    if constexpr (!traits::is_nullptr_v<decltype(grad)>) {
      // Direct accumulation of gradient and sparse Hessian:
      // r_0 = (x0 - t0) + 0.5 * (x2 - t2)
      // r_1 = (x1 - t1) + 0.4 * (x3 - t3)
      // r_2 = (x2 - t2)
      // r_3 = (x3 - t3)
      // r_4 = (x4 - t4) + 0.3 * (x1 - t1)
      std::vector<Eigen::Triplet<double>> triplets;
      // B (head 2)
      triplets.emplace_back(0, 0, 3.0);
      triplets.emplace_back(1, 1, 3.0);
      triplets.emplace_back(0, 1, 0.2);
      triplets.emplace_back(1, 0, 0.2);
      // E (coupling)
      triplets.emplace_back(0, 2, 0.5);
      triplets.emplace_back(2, 0, 0.5);
      triplets.emplace_back(1, 3, 0.4);
      triplets.emplace_back(3, 1, 0.4);
      triplets.emplace_back(1, 4, 0.3);
      triplets.emplace_back(4, 1, 0.3);
      // C (tail 3)
      triplets.emplace_back(2, 2, 4.0);
      triplets.emplace_back(3, 3, 4.0);
      triplets.emplace_back(4, 4, 4.0);
      triplets.emplace_back(2, 3, 0.1);
      triplets.emplace_back(3, 2, 0.1);

      H.resize(5, 5);
      H.setFromTriplets(triplets.begin(), triplets.end());

      grad = H * diff;
    }

    return 0.5 * diff.squaredNorm();
  };

  Options options(Options::Solver::GaussNewton);
  options.linear.method = LinearSolverMethod::LDLT;
  options.stop.max_iters = 10;

  SECTION("Optimization using Schur complement via reduced() trait") {
    CameraPointProblem<double> params;
    params.values << 0.0, 0.0, 0.0, 0.0, 0.0;

    gn::Optimizer<SparseMat> opt(options);
    const auto summary = opt.Optimize(params, loss);

    REQUIRE(summary.Succeeded());
    REQUIRE(summary.Converged());
    REQUIRE(opt.hasReduced());
    REQUIRE(opt.numReduced() == 2);

    for (int i = 0; i < 5; ++i) {
      REQUIRE(params.values(i) == Approx(target(i)).margin(1e-6));
    }

    // Regression: the covariance must invert the full Hessian, not the Schur-eliminated block,
    // which used to trip Eigen's SimplicialCholesky symbolic-analysis size assertion.
    const auto covariance = opt.Covariance();
    REQUIRE(covariance.has_value());
    const MatX expected_covariance = opt.Hessian().toDense().inverse();
    REQUIRE((covariance->toDense() - expected_covariance).norm() < 1e-10);
  }

  SECTION("Optimization using Levenberg-Marquardt with Schur complement") {
    CameraPointProblem<double> params;
    params.values << 0.0, 0.0, 0.0, 0.0, 0.0;

    Options lm_options(Options::Solver::LevenbergMarquardt);
    lm_options.linear.method = LinearSolverMethod::LDLT;
    lm_options.stop.max_iters = 15;

    lm::Optimizer<SparseMat> opt(lm_options);
    const auto summary = opt.Optimize(params, loss);

    REQUIRE(summary.Succeeded());
    REQUIRE(summary.Converged());
    REQUIRE(opt.hasReduced());
    REQUIRE(opt.numReduced() == 2);

    for (int i = 0; i < 5; ++i) {
      REQUIRE(params.values(i) == Approx(target(i)).margin(1e-5));
    }
  }

  SECTION("Optimization without reduced() trait uses standard sparse solver") {
    CameraPointProblemNoTrait<double> params;
    params.values << 0.0, 0.0, 0.0, 0.0, 0.0;

    gn::Optimizer<SparseMat> opt(options);
    const auto summary = opt.Optimize(params, loss);

    REQUIRE(summary.Succeeded());
    REQUIRE(summary.Converged());
    REQUIRE_FALSE(opt.hasReduced());
    REQUIRE(opt.numReduced() == 0);

    for (int i = 0; i < 5; ++i) {
      REQUIRE(params.values(i) == Approx(target(i)).margin(1e-6));
    }
  }

  SECTION("Schur complement optimization with locked parameters via zeroing") {
    CameraPointProblem<double> params;
    params.values << target(0), 0.0, 0.0, target(3), 0.0;

    gn::Optimizer<SparseMat> opt(options);
    // Lock index 0 (in reduced block) and index 3 (in eliminated block)
    opt.lock({0, 3});
    REQUIRE(opt.hasLocked());

    const auto summary = opt.Optimize(params, loss);
    REQUIRE(summary.Succeeded());
    REQUIRE(opt.hasReduced());
    REQUIRE_FALSE(opt.useLockedPermutation());

    // Locked parameters remain at initial locked values
    REQUIRE(params.values(0) == Approx(target(0)).margin(1e-9));
    REQUIRE(params.values(3) == Approx(target(3)).margin(1e-9));

    // Free parameters converge to target
    REQUIRE(params.values(1) == Approx(target(1)).margin(1e-5));
    REQUIRE(params.values(2) == Approx(target(2)).margin(1e-5));
    REQUIRE(params.values(4) == Approx(target(4)).margin(1e-5));
  }

  SECTION("Schur complement optimization logs reduced parameters") {
    CameraPointProblem<double> params;
    params.values << 0.0, 0.0, 0.0, 0.0, 0.0;

    Options log_options = options;
    log_options.log.enable = true;
    log_options.stop.max_iters = 5;

    gn::Optimizer<SparseMat> opt(log_options);
    const auto summary = opt.Optimize(params, loss);
    REQUIRE(summary.Succeeded());
    REQUIRE(opt.hasReduced());
    REQUIRE(opt.numReduced() == 2);
  }
}
