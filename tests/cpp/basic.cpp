// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <cmath>
#include <thread>

#if CATCH2_VERSION == 2
#include <catch2/catch.hpp>
#else
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#endif

#include <tinyopt/stop_reasons.h>
#include <tinyopt/tinyopt.h>

using namespace tinyopt;
using namespace tinyopt::nlls;

/// Common checks on an successful optimization
void SuccessChecks(const Summary &sum, StopReason expected_stop = StopReason::kMinError,
                   int min_num_iters = 2, int max_num_iters = 5) {
  REQUIRE(sum.Succeeded());
  REQUIRE(sum.num_iters >= min_num_iters);
  REQUIRE(sum.num_iters <= max_num_iters);
  if (min_num_iters > 0) {
    REQUIRE(sum.final_cost < 1e-5);
    REQUIRE(sum.Converged());
#if defined(TINYOPT_ENFORCE_NO_DYNAMIC_ALLOCATIONS)
    const std::size_t expected_history_size = sum.num_iters < 5 ? sum.num_iters : 5;
    REQUIRE(sum.hist.errs.size() == expected_history_size);
#else
    REQUIRE(sum.hist.errs.size() == size_t(sum.num_iters));
#endif
    REQUIRE(sum.hist.successes.size() == sum.hist.errs.size());
    REQUIRE(sum.hist.deltas2.size() == sum.hist.errs.size());
  }
#if !defined(TINYOPT_ENFORCE_NO_DYNAMIC_ALLOCATIONS)
  REQUIRE(sum.has_final_hessian());
  REQUIRE(sum.final_hessian_dense()(0, 0) > 0);
#endif
  REQUIRE(sum.stop_reason == expected_stop);
}

void TestSuccess() {
  // Normal case using LM
  {
    std::cout << "**** Normal Test Case LM \n";
    auto loss = [&](const auto &x, auto &grad, auto &H) {
      double res = x - 2;
      if constexpr (!traits::is_nullptr_v<decltype(grad)>) {
        H(0, 0) = 1;
        grad(0) = res;
      }
      return std::abs(res);
    };
    double x = 1;
    const auto &sum = Optimize(x, loss);
    SuccessChecks(sum, StopReason::kMinDeltaNorm);
  }
  {
    std::cout << "**** min || ||x-y|| + random || \n";
    const Vec2 y = 10 * Vec2::Random();  // prior
    auto loss = [&](const auto &x) {
      const auto res = (x - y).eval();
      return res.norm() + 0.1 * Vec1::Random()[0];
    };

    Vec2 x(5, 5);
    Options options;
    options.stop.max_iters = 10;
    options.lm.damping_init = 1e0;
    const auto &sum = Optimize(x, loss, options);
    REQUIRE(sum.Succeeded());
    REQUIRE(!sum.Converged());
  }
#if defined(TINYOPT_ENABLE_GAUSS_NEWTON)
  {
    std::cout << "**** Normal Test Case GN\n";
    auto loss = [&](const auto &x, auto &grad, auto &H) {
      double res = x - 2;
      if constexpr (!traits::is_nullptr_v<decltype(grad)>) {
        H(0, 0) = 1;
        grad(0) = res;
      }
      return std::abs(res);
    };
    double x = 1;
    Options options;
    options.solver_type = Options::Solver::GaussNewton;
    const auto &sum = Optimize(x, loss, options);
    SuccessChecks(sum);
  }
#endif
  // Timimg sum
  {
    std::cout << "**** Testing Time sum x\n";
    auto loss = [&](const auto &x, auto &grad, auto &H) {
      double res = x - VecXf::Random(1)[0];
      if constexpr (!traits::is_nullptr_v<decltype(grad)>) {
        H(0, 0) = VecXf::Random(1).cwiseAbs()[0];
        grad(0) = res;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      return std::abs(res);
    };
    double x = 0;
    Options options;
    options.stop.max_duration_ms = 5;
    options.stop.min_grad_norm2 = 0;  // disable
    const auto &sum = Optimize(x, loss, options);
    SuccessChecks(sum, StopReason::kTimedOut, 0);
  }
#if defined(TINYOPT_ENABLE_GAUSS_NEWTON)
  // Min error
  {
    std::cout << "**** Testing Minimum error\n";
    auto loss = [&](const auto &x, auto &grad, auto &H) {
      double res = x - 2;
      if constexpr (!traits::is_nullptr_v<decltype(grad)>) {
        H(0, 0) = 1;
        grad(0) = res;
      }
      return std::abs(res);
    };
    double x = 1;
    Options options;
    options.stop.min_error = 1e-2f;
    options.solver_type = Options::Solver::GaussNewton;
    const auto &sum = Optimize(x, loss, options);
    SuccessChecks(sum, StopReason::kMinError);
  }
#endif
  // User stop callback
  {
    std::cout << "**** User stop callback\n";
    auto loss = [&](const auto &x, auto &grad, auto &H) {
      double res = x - 2 + Vec1::Random()[0];
      if constexpr (!traits::is_nullptr_v<decltype(grad)>) {
        H(0, 0) = 1;
        grad(0) = res;
      }
      return std::abs(res);
    };
    double x = 1;
    Options options;
    options.stop.min_error = 0;
    options.stop.min_grad_norm2 = 0;
    options.stop.stop_callback2 = [](float, const VecXf &, const VecXf &g) {
      return g.norm() < 2.0;
    };
    const auto &sum = Optimize(x, loss, options);
    REQUIRE(sum.stop_reason == StopReason::kUserStopped);
  }

  // Step callback: summing the reported steps tracks the parameters
  {
    std::cout << "**** Step callback\n";
    Vec2 x(-1.2, 1.0);
    Vec2 tracked = x;
    int calls = 0;
    Options options;
    options.log.enable = false;
    const auto rosenbrock = [](const auto &x) {
      using T = std::decay_t<decltype(x[0])>;
      Eigen::Matrix<T, 2, 1> r;
      r << T(10.0) * (x[1] - x[0] * x[0]), T(1.0) - x[0];
      return r;
    };
    options.stop.step_callback = [&](const VecXf &dx, bool) {
      tracked += dx.cast<double>();
      ++calls;
      return false;
    };
    const auto &sum = Optimize(x, rosenbrock, options);
    REQUIRE(sum.Succeeded());
    REQUIRE(calls > 0);
    REQUIRE((tracked - x).norm() < 1e-3);

    // Returning true stops the optimization
    x = Vec2(-1.2, 1.0);
    options.stop.step_callback = [](const VecXf &, bool) { return true; };
    const auto &stopped = Optimize(x, rosenbrock, options);
    REQUIRE(stopped.stop_reason == StopReason::kUserStopped);
  }
}

/// Common checks on an early failure
void FailureChecks(const auto &sum, StopReason expected_stop = StopReason::kSolverFailed,
                   int max_iters = 1) {
  REQUIRE(!sum.Succeeded());
  REQUIRE(!sum.Converged());
  REQUIRE(sum.num_iters <= max_iters);  // can at most tried once
  REQUIRE(sum.hist.errs.empty());
  REQUIRE(sum.hist.successes.empty());
  REQUIRE(sum.hist.deltas2.empty());
  REQUIRE(sum.stop_reason == expected_stop);
}

void TestFailures() {
  // NaN in grad
  {
    std::cout << "**** Testing NaNs in Jt * res\n";
    auto loss = [&](const auto &x, auto &grad, auto &H) {
      double res = x - 2;
      if constexpr (!traits::is_nullptr_v<decltype(grad)>) {
        H(0, 0) = 1;
        grad(0) = NAN;  // a NaN? Yeah, that's bad NaN.
      }
      return std::abs(res);
    };
    double x = 1;
    const auto &sum = Optimize(x, loss);
    FailureChecks(sum, StopReason::kSystemHasNaNOrInf);
  }
  // Infinity in grad
  {
    std::cout << "**** Testing Infinity in grad\n";
    auto loss = [&](const auto &x, auto &grad, auto &H) {
      double res = x - 2;
      if constexpr (!traits::is_nullptr_v<decltype(grad)>) {
        H(0, 0) = 1;
        grad(0) = std::numeric_limits<double>::infinity();
      }
      return std::abs(res);
    };
    double x = 1;
    const auto &sum = Optimize(x, loss);
    FailureChecks(sum, StopReason::kSystemHasNaNOrInf);
  }
  // Infinity in grad
  {
    std::cout << "**** Testing Infinity in res\n";
    auto loss = [&](const auto &x, auto &grad, auto &H) {
      double res = x + std::numeric_limits<double>::infinity();
      if constexpr (!traits::is_nullptr_v<decltype(grad)>) {
        H(0, 0) = 1;
        grad(0) = std::numeric_limits<double>::infinity();
      }
      return std::abs(res);
    };
    double x = 1;
    const auto &sum = Optimize(x, loss);
    FailureChecks(sum, StopReason::kSystemHasNaNOrInf);
  }
  // Infinity in res*res
  {
    std::cout << "**** Testing Infinity in res\n";
    auto loss = [&](const auto &x, auto &grad, auto &H) {
      double res = x + 1;
      if constexpr (!traits::is_nullptr_v<decltype(grad)>) {
        H(0, 0) = 1;
        grad(0) = res;
      }
      return std::numeric_limits<double>::infinity();
    };
    double x = 1;
    const auto &sum = Optimize(x, loss);
    FailureChecks(sum, StopReason::kSystemHasNaNOrInf);
  }
  // Forgot to update H
  {
    std::cout << "**** Testing Forgot to update H\n";
    auto loss = [&](const auto &x, auto &, auto &) {
      double res = x - 2;
      // Let's forget to update gradient and hessian
      return std::abs(res);
    };
    double x = 1;
    Options options;
#if defined(TINYOPT_ENABLE_GAUSS_NEWTON)
    options.solver_type = Options::Solver::GaussNewton;
#endif
    options.hessian.check_min_H_diag = 1e-7f;
    const auto &sum = Optimize(x, loss, options);
    FailureChecks(sum, StopReason::kSolverFailed, 3);
  }
  // No residuals
  {
    std::cout << "**** No residuals\n";
    auto loss = [&](const auto &, auto &, auto &) {
      return VecX();  // no residuals
    };
    double x = 1;
    const auto &sum = Optimize(x, loss);
    FailureChecks(sum, StopReason::kSkipped);
  }
  // Empty x
  {
    std::cout << "**** Testing Empty x\n";
    auto loss = [&](const auto &x, auto &grad, auto &H) {
      float res = x[0] - 2;
      if constexpr (!traits::is_nullptr_v<decltype(grad)>) {
        H(0, 0) = 1;
        grad(0) = res;
      }
      return std::abs(res);
    };
    std::vector<float> empty;
    const auto &sum = Optimize(empty, loss);
    FailureChecks(sum, StopReason::kSkipped);
  }
// Out of memory (only on linux, not sure why it crashes on MacOS..)
#if (defined(LINUX) || defined(__linux__))
  {
    std::cout << "**** Testing Out of Memory x\n";
    auto loss = [&](const auto &x, auto &grad, auto &H) {
      double res = x[0] - 2;
      if constexpr (!traits::is_nullptr_v<decltype(grad)>) {
        H(0, 0) = 1;
        grad(0) = res;
      }
      return std::abs(res);
    };
    std::vector<double> too_large;
    try {
      // unless you're Elon and can afford that memoryfor a dense H matrix
      too_large.resize(100000);
      const auto &sum = Optimize(too_large, loss);
      FailureChecks(sum, StopReason::kOutOfMemory);
    } catch (const std::bad_alloc &e) {
      std::cout << "CAN'T EVEN ALLOCATE x...\n";
    }
  }
#endif
}

TEST_CASE("tinyopt_basic_success") { TestSuccess(); }

TEST_CASE("tinyopt_basic_failures") { TestFailures(); }

#if !defined(TINYOPT_ENABLE_GAUSS_NEWTON)
TEST_CASE("disabled GaussNewton cannot be selected") {
  auto loss = [](const auto &params, auto &gradient, auto &hessian) {
    const auto residual = params - 2.0;
    if constexpr (!traits::is_nullptr_v<decltype(gradient)>) {
      gradient(0) = residual;
      hessian(0, 0) = 1.0;
    }
    return residual * residual;
  };
  double x = 0.0;
  Options options(Options::Solver::GaussNewton);
  REQUIRE_THROWS(Optimize(x, loss, options));
}
#endif