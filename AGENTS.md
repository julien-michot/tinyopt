# AGENTS.md — Development Guidelines for Tinyopt

Welcome to `Tinyopt`, a high-performance, header-only C++20 optimization library engineered for unconstrained optimization and non-linear least squares (NLLS) problems.

As an AI agent or engineer working in this repository, you **must** uphold the highest engineering standards: zero compiler warnings, zero dynamic memory allocations in critical paths, rigorous mathematical and gradient verification, and consistent code formatting.

---

## 1. Repository Architecture & Core Philosophy

Tinyopt achieves superior computational speed and memory efficiency through its **Accumulation Pattern**:
- Unlike traditional NLLS libraries that allocate and buffer large arrays of residual vectors and Jacobian matrices, `Tinyopt` empowers users and solvers to accumulate gradients ($J^T r$) and Hessian approximations ($J^T J$) directly into the linear system.
- This curtailment of memory allocation minimizes cache misses and unlocks rapid convergence for small-to-medium and structured optimization problems.

### Directory Layout

```text
tinyopt/
├── include/tinyopt/          # Header-only core library
│   ├── tinyopt.h             # Umbrella include header
│   ├── traits.h              # Template metaprogramming & type traits
│   ├── types.h               # Eigen aliases (VecX, MatX, etc.) & container types
│   ├── cost.h                # Cost evaluation and residual wrappers
│   ├── optimize.h            # High-level entry point (Optimize templates)
│   ├── stop_reasons.h        # Convergence and termination criteria
│   ├── time.h                # High-resolution profiling utilities
│   ├── optimizers/           # Iterative optimization algorithms
│   │   ├── optimizer.h       # Base iterative loop, trust region / step logic
│   │   ├── lm.h              # Levenberg-Marquardt optimizer
│   │   ├── gn.h              # Gauss-Newton optimizer
│   │   ├── gd.h              # Gradient Descent optimizer
│   │   └── options.h         # Solver options & parameters
│   ├── solvers/              # Linear system solvers
│   │   ├── lm.h, gn.h, gd.h  # Linear step solvers
│   │   └── base.h            # Solver base interface
│   ├── diff/                 # Differentiation mechanisms
│   │   ├── auto_diff.h       # Automatic differentiation (Jet-based)
│   │   ├── jet.h             # Dual numbers / Jet class implementation
│   │   ├── num_diff.h        # Finite-difference numerical differentiation
│   │   └── gradient_check.h  # Mathematical derivative verification utilities
│   ├── losses/               # Loss functions & M-estimators
│   │   ├── norms.h           # L1, L2, squared L2
│   │   ├── robust_norms.h    # Huber, Cauchy, Tukey, etc.
│   │   └── activations.h     # Activation functions
│   └── 3rdparty/             # Adapters for Sophus, Lie++, Ceres
├── tests/                    # Catch2 v3 unit test suite
├── benchmarks/               # Performance benchmarks (Catch2 & Ceres comparison)
├── examples/                 # Real-world usage examples (gravitational lensing, triangulation)
├── cmake/                    # Modular CMake configuration files
├── pixi.toml                 # Pixi environment & dependency manager
└── .clang-format             # Code formatting rules (2-space, Google-based)
```

---

## 2. High-Performance C++20 Standards

Tinyopt is built for speed. Every line of C++ code must satisfy these high-performance principles:

1. **Zero Allocations in Inner Optimization Loops**:
   - Never call `malloc`, `new`, `std::vector::resize`, or create dynamic Eigen matrices (`MatrixXd`, `VectorXd`) inside cost evaluations, residual evaluations, or solver iteration loops.
   - For fixed-size problems (e.g. 2D/3D points, poses, camera parameters), use fixed-size Eigen types (`Vector<T, N>`, `Matrix<T, Rows, Cols>`, `Vec2`, `Vec3`, `Mat33`).
   - If dynamic-size systems are required, allocate buffers once prior to the optimization loop and pass them by reference or reuse solver workspace memory.

2. **Eigen Expression Templates & Aliasing**:
   - Avoid hidden temporaries when multiplying matrices. Use `.noalias()` when assigning matrix products to an lvalue where operands do not overlap:
     ```cpp
     // CORRECT:
     H.noalias() += J.transpose() * J;
     g.noalias() += J.transpose() * res;

     // INCORRECT (creates temporary matrix):
     H += J.transpose() * J;
     ```
   - Avoid unnecessary `.eval()` calls unless aliasing is unavoidable.

3. **C++20 Idioms & Compile-Time Dispatch**:
   - Use `if constexpr` to eliminate branching at runtime for type-dependent operations (e.g., checking if gradient/Hessian output pointers are `nullptr`).
   - Leverage `traits::` (e.g., `traits::is_nullptr_v<decltype(grad)>`, `traits::is_jet_v<T>`).
   - Mark functions `constexpr` and `inline` whenever feasible.
   - Mark non-mutating methods and getters `const` and `[[nodiscard]]`.

4. **Cache Friendliness & Data Locality**:
   - Access matrices in column-major order (Eigen default) in tight loops.
   - Keep parameter blocks compact and contiguous in memory.

---

## 3. Strict Compiler & Code Quality Policies

### Zero Compiler Warnings (`-Werror`)
- Both GCC and Clang build with `-Wall -Wextra -Werror`.
- **No warning will be tolerated.** A single warning breaks the build.
- Do not suppress warnings with `#pragma GCC diagnostic ignored` unless it is an external header issue and well-documented.

### Code Style & Formatting
- **Clang-Format**: All code must conform to the repository's `.clang-format` (Google-based, 2 spaces indentation, 100 character line limit).
- Run `pixi run -e fmt clang-format -i <file>` or `./.agents/skills/tinyopt-dev-workflow/scripts/format.sh` before committing changes.

### License & Copyright Header
Every new or modified C++ source file (`.h`, `.cpp`) **must** begin with the official license header:
```cpp
// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0
```

---

## 4. Testing & Verification Requirements

No code is complete without exhaustive testing.

1. **Catch2 v3 Test Suite**:
   - All tests live under `tests/` and use Catch2 v3 (`<catch2/catch_test_macros.hpp>`, `<catch2/catch_approx.hpp>`, `<catch2/generators/catch_generators.hpp>`).
   - Use `Approx(...).margin(...)` or `.epsilon(...)` with reasonable numeric bounds. Never use loose tolerances that could mask bugs.

2. **Mandatory Derivative Verification**:
   - Whenever writing or modifying an optimization problem, residual, or loss function, you **must** verify the analytical derivatives against numerical derivatives using `diff::CheckResidualsGradient` or `diff::CheckCostGradient`:
     ```cpp
     REQUIRE(diff::CheckResidualsGradient(x0, residuals));
     ```
   - Test automatic differentiation (Jets) alongside analytical gradients to ensure consistency.

3. **Convergence & Optimality Verification**:
   - In optimizer tests, assert that:
     ```cpp
     REQUIRE(out.Succeeded());
     REQUIRE(out.Converged());
     ```
   - Check the final parameter error: `std::abs(x - ground_truth) < tolerance`.
   - Check first-order optimality: final gradient norm $||\nabla f(x^*)||$ must be close to zero.

4. **Edge Cases & Numerical Stability**:
   - Test bad initial conditions (away from the basin of attraction).
   - Test zero gradient points, saddle points, and ill-conditioned Hessians.
   - Verify that non-finite values (NaN, Inf) are handled gracefully and trigger the appropriate `StopReason`.

5. **AddressSanitizer (ASAN)**:
   - Run tests under ASAN to ensure zero memory leaks, buffer overflows, or use-after-scope errors:
     ```shell
     cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=ASAN -DTINYOPT_BUILD_TESTS=ON
     cmake --build build
     cd build && ctest --output-on-failure
     ```

---

## 5. Development Workflow & Commands

The project uses [Pixi](https://pixi.prefix.dev/) to manage dependencies and build environments.

### Environments
- `test`: Catch2, Sophus, GCC/Clang (for running the test suite).
- `bench`: Benchmarking tools, Ceres Solver, OpenMP.
- `all`: All optional dependencies.

### Common Commands

```shell
# 1. Clean build directory
pixi run clean

# 2. Configure for testing
pixi run configure-test

# 3. Build test suite
pixi run -e test build

# 4. Run all unit tests
pixi run -e test test
# Or directly via ctest:
cd build && ctest --output-on-failure

# 5. Run an individual test binary with full Catch2 output:
./build/tests/tinyopt_test_optimize_easy -s

# 6. Run benchmarks:
pixi run -e bench bench
```

> **Important**: When switching between Pixi environments (e.g. from `test` to `bench`), always clean `build/` first (`pixi run clean`) to avoid CMake cache collisions between different conda prefixes.

---

## 6. Skills Available in `.agents/skills/`

Antigravity provides specialized skills to assist with tinyopt development:
- **`tinyopt-dev-workflow`**: Step-by-step commands to configure, build, format, and debug tests.
- **`tinyopt-testing-and-validation`**: Test writing guide, derivative validation rules, Catch2 v3 templates, and edge case checklist.
- **`tinyopt-perf-and-architecture`**: Deep-dive into zero-allocation programming, Eigen expression templates, and optimization loop profiling.
