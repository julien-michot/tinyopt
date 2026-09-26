# AGENTS.md — Development Guidelines for Tinyopt

Welcome to `Tinyopt`, a high-performance, header-only C++20 optimization library engineered for unconstrained optimization and non-linear least squares (NLLS) problems.

As an AI agent or engineer working in this repository, you **must** uphold the highest engineering standards: zero compiler warnings, zero dynamic memory allocations in critical paths, rigorous mathematical and gradient verification, and consistent code formatting.

> **CRITICAL AGENT COMMIT POLICY**:
> AI agents must **NEVER** automatically create git commits (`git commit`) unless explicitly commanded to do so by the user within the active session (e.g., "ok commit now", "create a commit"). Keep working changes in the working tree or staging area.

---

## 1. Repository Architecture & Core Philosophy

Tinyopt achieves superior computational speed and memory efficiency through its **Accumulation Pattern**:
- Unlike traditional NLLS libraries that allocate and buffer large arrays of residual vectors and Jacobian matrices, `Tinyopt` empowers users and solvers to accumulate gradients ($J^T r$) and Hessian approximations ($J^T J$) directly into the linear system.
- This curtailment of memory allocation minimizes cache misses and unlocks rapid convergence for small-to-medium and structured optimization problems.
- For a comprehensive architectural deep-dive, see [docs/architecture.md](docs/architecture.md).

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
├── docs/                     # Documentation (architecture, style, guidelines, API)
├── cmake/                    # Modular CMake configuration files
├── pixi.toml                 # Pixi environment & dependency manager
└── .clang-format             # Code formatting rules (2-space, Google-based)
```

---

## 2. Basic Software Development Guidelines

1. **KISS & Readability First**: Optimization mathematics can be complex. Avoid speculative abstraction; keep implementations clear, concise, and mathematically self-evident.
2. **Single Responsibility Principle (SRP)**:
   - A *loss function* evaluates scalar metrics and its derivatives.
   - A *step solver* computes the linear update step $\delta x$.
   - An *optimizer* governs the trust region, damping parameter $\lambda$, step acceptance, and stopping criteria.
3. **Defensive Numerical Programming**:
   - Check for non-finite values (`std::isnan`, `std::isinf`) in gradients and Hessians.
   - Always map numerical breakdown to a clean [StopReason](include/tinyopt/stop_reasons.h) rather than producing undefined behavior or crashes.
4. **Zero Compiler Warnings**: No warning will be tolerated under `-Wall -Wextra -Werror`.
5. For full guidelines, see [docs/development_guidelines.md](docs/development_guidelines.md).

---

## 3. High-Performance C++20 & Coding Style

For the complete coding style guide, see [docs/coding_style.md](docs/coding_style.md).

1. **Zero Allocations in Inner Optimization Loops**:
   - Never call `malloc`, `new`, `std::vector::resize`, or create dynamic Eigen matrices (`MatrixXd`, `VectorXd`) inside cost evaluations, residual evaluations, or solver iteration loops.
   - For fixed-size problems, use fixed-size Eigen types (`Vector<T, N>`, `Matrix<T, Rows, Cols>`, `Vec2`, `Vec3`, `Mat33`).
   - If dynamic-size systems are required, allocate buffers once prior to the optimization loop and pass them by reference.

2. **Eigen Expression Templates & Aliasing**:
   - Avoid hidden temporaries when multiplying matrices. Always use `.noalias()` when assigning matrix products to an lvalue where operands do not overlap:
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

## 4. Strict Compiler & Code Quality Policies

### Zero Compiler Warnings (`-Werror`)
- Both GCC and Clang build with `-Wall -Wextra -Werror`.
- A single warning breaks the build.
- Do not suppress warnings with `#pragma GCC diagnostic ignored`.

### Code Style & Formatting
- **Clang-Format**: All code must conform to the repository's `.clang-format` (Google-based, 2 spaces indentation, 100 character line limit).
- Run `./.agents/skills/tinyopt-dev-workflow/scripts/format.sh` before committing changes.

### License & Copyright Header
Every new or modified C++ source file (`.h`, `.cpp`) **must** begin with the official license header:
```cpp
// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0
```

---

## 5. Testing & Verification Requirements

No code is complete without exhaustive testing.

1. **Catch2 v3 Test Suite**:
   - All tests live under `tests/` and use Catch2 v3 (`<catch2/catch_test_macros.hpp>`, `<catch2/catch_approx.hpp>`, `<catch2/generators/catch_generators.hpp>`).
   - Use `Approx(...).margin(...)` or `.epsilon(...)` with reasonable numeric bounds.

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

4. **AddressSanitizer (ASAN)**:
   - Run tests under ASAN to ensure zero memory leaks, buffer overflows, or use-after-scope errors:
     ```shell
     cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=ASAN -DTINYOPT_BUILD_TESTS=ON
     cmake --build build
     cd build && ctest --output-on-failure
     ```

---

## 6. Git Commit Conventions (With Emojis)

When instructed by the user to commit, all commit titles **must** use the following emoji convention:

```text
<emoji> <type>(<optional-scope>): <subject>
```

| Emoji | Type | Description |
| :---: | :--- | :--- |
| 📝 | `docs` | Documentation only changes |
| ✨ | `feat` | New algorithms, solvers, features, or APIs |
| 🐛 | `fix` | Bug fixes and numerical stabilization |
| ⚡ | `perf` | Performance optimizations, zero-allocation improvements |
| 🧪 | `test` | Adding or modifying Catch2 tests and benchmarks |
| ♻️ | `refactor` | Code restructuring without behavioral changes |
| 🎨 | `style` | Formatting, clang-format adjustments, whitespace |
| 🔧 | `chore` | Tooling, Pixi dependencies, CMake updates |
| 🔒 | `security` | Sanitizer fixes, bounds checking, vulnerability fixes |

---

## 7. Development Workflow & Commands

The project uses [Pixi](https://pixi.prefix.dev/) to manage dependencies and build environments.

### Environments
- `test`: Catch2, Sophus, GCC/Clang (for running the test suite).
- `bench`: Benchmarking tools, Ceres Solver, OpenMP.
- `all`: All optional dependencies.

### Common Commands

```shell
# 1. Run all tests via helper script (automatically handles pixi test env)
./.agents/skills/tinyopt-dev-workflow/scripts/run_tests.sh

# 2. Run a specific test with verbose Catch2 output
./.agents/skills/tinyopt-dev-workflow/scripts/run_tests.sh tinyopt_test_sqrt2

# 3. Format all code according to .clang-format
./.agents/skills/tinyopt-dev-workflow/scripts/format.sh

# 4. Dry-run format check
./.agents/skills/tinyopt-dev-workflow/scripts/format.sh --check

# 5. Clean build directory
pixi run clean
```

> **Important**: When switching between Pixi environments (e.g. from `test` to `bench`), always clean `build/` first (`pixi run clean`) to avoid CMake cache collisions between different conda prefixes.

---

## 8. Skills Available in `.agents/skills/`

Antigravity provides specialized skills to assist with tinyopt development:
- **`tinyopt-dev-workflow`**: Step-by-step commands to configure, build, format, and debug tests.
- **`tinyopt-testing-and-validation`**: Test writing guide, derivative validation rules, Catch2 v3 templates, and edge case checklist.
- **`tinyopt-perf-and-architecture`**: Deep-dive into zero-allocation programming, Eigen expression templates, and optimization loop profiling.
