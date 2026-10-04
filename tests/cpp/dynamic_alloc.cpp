// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#ifndef EIGEN_RUNTIME_NO_MALLOC
#define EIGEN_RUNTIME_NO_MALLOC
#endif

#include <cstdio>
#include <cstdlib>
#include <new>

#include <Eigen/Eigen>

#if CATCH2_VERSION == 2
#include <catch2/catch.hpp>
#else
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#endif

#include <tinyopt/optimize.h>
#include <tinyopt/tinyopt.h>

namespace {

bool track_allocations = false;
std::size_t allocation_count = 0;

class AllocationTracking {
 public:
  AllocationTracking() {
    allocation_count = 0;
    track_allocations = true;
  }
  ~AllocationTracking() { track_allocations = false; }

  std::size_t Count() const { return allocation_count; }
};

class EigenMallocDisallowed {
 public:
  EigenMallocDisallowed() : previous_(Eigen::internal::is_malloc_allowed()) {
    Eigen::internal::set_is_malloc_allowed(false);
  }
  ~EigenMallocDisallowed() { Eigen::internal::set_is_malloc_allowed(previous_); }

 private:
  bool previous_;
};

}  // namespace

void *operator new(std::size_t size) {
  void *memory = std::malloc(size == 0 ? 1 : size);
  if (memory == nullptr) throw std::bad_alloc();
  if (track_allocations) ++allocation_count;
  return memory;
}

void *operator new[](std::size_t size) {
  void *memory = std::malloc(size == 0 ? 1 : size);
  if (memory == nullptr) throw std::bad_alloc();
  if (track_allocations) ++allocation_count;
  return memory;
}

void operator delete(void *memory) noexcept { std::free(memory); }
void operator delete[](void *memory) noexcept { std::free(memory); }
void operator delete(void *memory, std::size_t) noexcept { std::free(memory); }
void operator delete[](void *memory, std::size_t) noexcept { std::free(memory); }

#if defined(TINYOPT_ENFORCE_NO_DYNAMIC_ALLOCATIONS)
TEST_CASE("tinyopt_strict_history_keeps_first_and_latest_values") {
  tinyopt::Summary::History history;
  history.Add(1.0, 2.0, true);
  history.Add(3.0, 4.0, false);
  history.Add(5.0, 6.0, true);

  REQUIRE(history.size == 2);
  REQUIRE(history.errs[0] == 1.0);
  REQUIRE(history.errs[1] == 5.0);
  REQUIRE(history.deltas2[0] == 2.0);
  REQUIRE(history.deltas2[1] == 6.0);
  REQUIRE(history.successes[0]);
  REQUIRE(history.successes[1]);
  REQUIRE(history.last_delta2() == 6.0);
}
#endif

TEST_CASE("tinyopt_fixed_size_manual_accumulation_avoids_eigen_allocations") {
  tinyopt::Vec2f x = tinyopt::Vec2f::Zero();
  const tinyopt::Vec2f target(1.0f, -2.0f);
  tinyopt::Options options;
  options.log.enable = false;
#if !defined(TINYOPT_ENFORCE_NO_DYNAMIC_ALLOCATIONS)
  options.hessian.save_last = false;
#endif

  std::size_t allocations = 0;
  std::size_t ordinary_residual_allocations = 0;
  tinyopt::Summary sum;
  {
    AllocationTracking tracking;
    {
      EigenMallocDisallowed no_eigen_malloc;
      sum = tinyopt::Optimize(
          x,
          [&](const auto &value, auto &gradient, auto &hessian) {
            const std::size_t before = allocation_count;
            const auto residual = value - target;
            if constexpr (!tinyopt::traits::is_nullptr_v<decltype(gradient)>) {
              gradient = 2.0f * residual;
              hessian = 2.0f * tinyopt::Mat2f::Identity();
            }
            ordinary_residual_allocations += allocation_count - before;
            return residual.squaredNorm();
          },
          options);
    }
    allocations = tracking.Count();
  }

  std::printf("manual Vec2f Optimize: %zu ordinary allocations, %zu in accumulation\n", allocations,
              ordinary_residual_allocations);
  REQUIRE(sum.Succeeded());
  REQUIRE(sum.Converged());
  REQUIRE(ordinary_residual_allocations == 0);
#if defined(TINYOPT_ENFORCE_NO_DYNAMIC_ALLOCATIONS)
  REQUIRE(allocations == 0);
  REQUIRE_FALSE(sum.has_final_hessian());
#endif
  REQUIRE((x - target).norm() < 1e-5f);
}

TEST_CASE("tinyopt_variadic_manual_accumulation_avoids_eigen_allocations") {
  double s = 0.0;
  tinyopt::Vec3 x = tinyopt::Vec3::Zero();
  tinyopt::Options options;
  options.log.enable = false;
#if !defined(TINYOPT_ENFORCE_NO_DYNAMIC_ALLOCATIONS)
  options.hessian.save_last = false;
#endif

  std::size_t allocations = 0;
  tinyopt::Summary sum;
  {
    AllocationTracking tracking;
    {
      EigenMallocDisallowed no_eigen_malloc;
      sum = tinyopt::Optimize(
          s, x,
          [](const auto &scale, const auto &value, auto &gradient, auto &hessian) {
            Eigen::Matrix<typename std::decay_t<decltype(value)>::Scalar, 4, 1> residual;
            residual(0) = scale - 1.0;
            residual.template tail<3>() = value - tinyopt::Vec3::Ones();
            if constexpr (!tinyopt::traits::is_nullptr_v<decltype(gradient)>) {
              gradient = residual;
              hessian.setIdentity();
            }
            return residual.squaredNorm();
          },
          options);
    }
    allocations = tracking.Count();
  }

  std::printf("variadic manual accumulation Optimize(s, x, acc): %zu ordinary allocations\n",
              allocations);
  REQUIRE(sum.Succeeded());
  REQUIRE(sum.Converged());
#if defined(TINYOPT_ENFORCE_NO_DYNAMIC_ALLOCATIONS)
  REQUIRE(allocations == 0);
  REQUIRE_FALSE(sum.has_final_hessian());
#else
  REQUIRE(allocations == 3);
#endif
  REQUIRE(s == Catch::Approx(1.0).margin(1e-5));
  REQUIRE((x - tinyopt::Vec3::Ones()).norm() < 1e-5);
}

TEST_CASE("tinyopt_variadic_autodiff_reports_allocations") {
  double s = 1.0;
  tinyopt::Vec3 x = tinyopt::Vec3::Zero();
  tinyopt::Options options;
  options.log.enable = false;
#if !defined(TINYOPT_ENFORCE_NO_DYNAMIC_ALLOCATIONS)
  options.hessian.save_last = false;
#endif
  std::size_t allocations = 0;
  std::size_t ordinary_residual_allocations = 0;
  tinyopt::Summary sum;
  {
    AllocationTracking tracking;
    {
      EigenMallocDisallowed no_eigen_malloc;
      sum = tinyopt::Optimize(
          s, x,
          [&](const auto &scale, const auto &value) {
            const std::size_t before = allocation_count;
            using Scalar = std::decay_t<decltype(value(0))>;
            Eigen::Matrix<Scalar, 4, 1> residual;
            residual.template head<3>() =
                scale * value - Eigen::Matrix<Scalar, 3, 1>::Constant(Scalar(1.0));
            residual(3) = scale - Scalar(1.0);
            ordinary_residual_allocations += allocation_count - before;
            return residual;
          },
          options);
    }
    allocations = tracking.Count();
  }

  std::printf(
      "variadic autodiff Optimize(s, x, residuals): %zu ordinary allocations, "
      "%zu ordinary allocations in residual evaluation; Eigen malloc disabled\n",
      allocations, ordinary_residual_allocations);
  REQUIRE(sum.Succeeded());
  REQUIRE(sum.Converged());
#if defined(TINYOPT_ENFORCE_NO_DYNAMIC_ALLOCATIONS)
  REQUIRE(allocations == 0);
  REQUIRE_FALSE(sum.has_final_hessian());
#else
  REQUIRE(allocations == 3);
#endif
  REQUIRE(ordinary_residual_allocations == 0);
  REQUIRE(s == Catch::Approx(1.0).margin(1e-5));
  REQUIRE((x - tinyopt::Vec3::Ones()).norm() < 1e-5);
}