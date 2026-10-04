# AGENTS.md — Development Guidelines for Tinyopt

Welcome to `Tinyopt`, a high-performance, header-only C++20 optimization library engineered for unconstrained optimization and non-linear least squares (NLLS) problems.

As an AI agent or engineer working in this repository, you **must** uphold the highest engineering standards: zero compiler warnings, zero dynamic memory allocations in critical paths, rigorous mathematical and gradient verification, and consistent code formatting.

> **CRITICAL AGENT COMMIT POLICY**:
> AI agents must **NEVER** automatically create git commits (`git commit`) unless explicitly commanded to do so by the user within the active session (e.g., "ok commit now", "create a commit"). Keep working changes in the working tree or staging area.

---

## Refactorings and Experiments

Before any refactoring or code experiment, read [docs/refactoring_and_experiment_guidelines.md](docs/refactoring_and_experiment_guidelines.md)
and follow its baseline, validation, and result-recording workflow. For every code change under
`include/` or `src/`, run `pixi run refactor-timings` as the final validation step after all edits;
it measures clean test-target compilation and non-Ceres benchmark runtime against the current
commit's saved timing results. After committing, run `pixi run refactor-timings --finalize` to
associate the `dev` result with the new commit hash.

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
├── tests/
│   ├── cpp/                  # Catch2 C++ unit tests
│   └── c/                    # C ABI tests and generated fixed-size C tests
├── benchmarks/               # Performance benchmarks (Catch2 & Ceres comparison)
├── examples/                 # Real-world usage examples (gravitational lensing, triangulation)
├── docs/                     # Documentation (architecture, style, guidelines, API, releases)
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
4. **Regression Tests are Mandatory**:
   - If a solver, optimizer, loss, matrix backend, or autodiff path breaks or regresses, add a dedicated regression test covering the failing mode before shipping.
   - For backend-specific bugs (dense vs. sparse, CPU vs. macOS, robust losses vs. plain residuals), include a minimal reproducer in the relevant test file and keep it narrow but exact to the failure.
   - Do not close a bugfix by only changing production code; the failing scenario must be exercised by at least one test.
   - When public API behavior changes, update the relevant examples in [docs/API.md](docs/API.md), [docs/tutorial.md](docs/tutorial.md), and the compile-check coverage in `tests/cpp/tutorial.cpp` so the docs remain in sync with the code.
   - When changing a C++ public interface, update every language binding that exposes it (including the C API and Python/other bindings where applicable), along with binding-specific tests and documentation. Do not leave a binding with a stale signature or behavior.
5. **Zero Compiler Warnings**: No warning will be tolerated under `-Wall -Wextra -Werror`.
6. For full guidelines, see [docs/development_guidelines.md](docs/development_guidelines.md).

---

## 3. High-Performance C++20 & Coding Style

For the complete coding style guide, see [docs/coding_style.md](docs/coding_style.md).

1. **Zero Allocations in Inner Optimization Loops**:
   - Never call `malloc`, `new`, `std::vector::resize`, or create dynamic Eigen matrices (`MatrixXd`, `VectorXd`) inside cost evaluations, residual evaluations, or solver iteration loops.
   - For fixed-size problems, use fixed-size Eigen types (`Vector<T, N>`, `Matrix<T, Rows, Cols>`, `Vec2`, `Vec3`, `Mat33`).
   - If dynamic-size systems are required, allocate buffers once prior to the optimization loop and pass them by reference.
   - Keep fixed-size Tinyopt paths allocation-free. Validate them with `-DTINYOPT_ENFORCE_NO_DYNAMIC_ALLOCATIONS=ON`; strict mode keeps only the latest five history samples, omits the final Hessian, disables optimizer logging, and guards Eigen against heap allocation. This applies only to fixed-size parameter types; residuals, accumulation functions, and callbacks must also avoid heap allocation.

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
- Run `./scripts/format.sh` before committing changes.
- Formatting-only changes do not require rebuilding or rerunning tests. After a formatter or commit hook rewrites code, inspect the diff and run the formatting check; rerun builds or tests only if the changes are not whitespace-only or reveal a substantive issue.

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

4. **Performance Regression Checks**:
   - Refactors and algorithm changes must verify benchmarks before and after the change.
   - Store or compare against a local baseline in `benchmarks/` or a small reproducible timing snapshot so performance can be checked in CI or locally.
   - For any optimization or solver change, show the benchmark delta and summarize whether the result is faster, slower, or within noise.
   - If a change makes performance materially worse, do not merge it without a compelling reason and an explicit user decision.

5. **Feature Coverage Requirement**:
   - Every new solver, loss function, optimizer mode, or user-facing capability must come with at least one dedicated test and at least one benchmark or micro-benchmark covering the common use case.
   - New public APIs should be exercised in both a correctness test and a performance test.

6. **Public API Simplicity**:
   - Prefer a minimal, high-level interface for common workflows.
   - Do not add configuration surface area unless there is a demonstrated need.
   - Keep the high-level entry points simple and ergonomic; advanced knobs should stay behind clearly named option structs or specialized overloads.

7. **Decision Suggestion Log**:
   - When a technical decision is non-trivial (API shape, solver behavior, benchmark trade-offs, naming, or new feature scope), produce a short list of options with a recommendation and ask the user which direction to adopt.
   - Examples of decisions to surface: solver selection defaults, new default tolerances, naming conventions, public API exposure, benchmark policy, and compatibility trade-offs.
   - Use the following template when a choice matters:
     ```text
     Decision: [short name]
     Option A: [description]
     Option B: [description]
     Option C: [description]
     Recommendation: [preferred option]
     Why: [brief rationale]
     User decision requested: yes/no
     ```

8. **Benchmark-First Refactor Rule**:
   - For every refactor or performance-sensitive change, capture or compare against a local benchmark baseline before and after the change.
   - Summarize the result in the final agent report as: faster, slower, or neutral within noise, with the actual measured delta.
   - Do not silently trade performance for readability without explicit approval.

9. **No Silent API Growth**:
   - New public interfaces must justify their need and keep the common workflow simple.
   - If a new knob is added, prefer a clearly named option struct or specialized overload instead of broadening the default call surface.
   - If the API becomes harder to understand, treat it as a design problem, not as a checkbox task.

10. **Feature Parity Requirement**:
    - Any new solver, optimizer mode, loss function, or user-facing capability must include:
      - a correctness test,
      - a convergence or stability test when relevant,
      - a benchmark or micro-benchmark,
      - a short example or docs mention if it is user-facing.
    - Do not add a feature “half-finished” without the matching validation work.

11. **AddressSanitizer (ASAN)**:
   - Run tests under ASAN to ensure zero memory leaks, buffer overflows, or use-after-scope errors:
     ```shell
     cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=ASAN -DTINYOPT_BUILD_TESTS=ON
     cmake --build build
     cd build && ctest --output-on-failure
     ```

---

## 6. Git Commit Conventions (With Emojis)

When instructed by the user to commit, all commit titles **must** use the following emoji convention and include a short body explaining the change.

Add a small code example to every code-related feature or API change, and make sure the example meaningfully clarifies the new behavior. Do not force a generic snippet like `auto x = tinyopt::Optimize(x0, residuals);` into every commit when it adds no value; for user-facing changes, prefer a minimal real-world example such as a residual tuple or solver call that demonstrates the new API.

```text
<emoji> <type>(<optional-scope>): <subject>

<body>
```

Example:
```text
✨ feat(solver): add Levenberg-Marquardt damping

This improves robustness on ill-conditioned problems by scaling the step
with an adaptive damping factor.
```

When `pixi run refactor-timings` has produced a comparison, include its two metrics in one concise
commit-body line. Use the measured values and status; omit this line if no valid baseline exists.
For example:
```text
Timing: test compile 43.2 s (-8%);
Benchmark mean 4 ms (+1%, neutral).
```
Treat benchmark changes within 5% as neutral, consistent with the timing task's report. Do not
round away a material regression or claim a comparison that was not measured.

When a code sample is useful, keep it minimal and directly relevant:
```cpp
auto step = solver.ComputeStep(x, residuals);
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

### PR Output Format
When asked to draft or create a pull request, do both of the following:
1. Show the PR title and body in normal readable Markdown for the user to review.
2. Then show a separate fenced `markdown` code block containing only the PR description text, labeled: `Here is the code block of the PR description (only) you can paste manually.`

Preserve headings, lists, and nested code fences exactly in the code block. Do not provide a raw PR description without the code block. This prevents copy/paste from stripping formatting.

Include a code sample only if it genuinely clarifies the public behavior; otherwise keep the PR text prose-only.

Example:
```markdown
## PR title

✨ chore(repo): standardize project scripts and task flow

## PR description

### Summary
This PR standardizes the project’s local development workflow and makes the repo checks reproducible across environments.
```

Here is the code block of the PR description (only) you can paste manually.

```markdown
## PR description

### Summary
This PR standardizes the project’s local development workflow and makes the repo checks reproducible across environments.
```

---

## 7. Development Workflow & Commands

For package generation and the version-bump commit, packaging, and tagging sequence, see
[docs/packaging_and_releasing.md](docs/packaging_and_releasing.md).

The project uses [Pixi](https://pixi.prefix.dev/) to manage dependencies and build environments.

### Environments
- `test`: Catch2, Sophus, GCC/Clang (for running the test suite).
- `bench`: Benchmarking tools, Ceres Solver, OpenMP.
- `all`: All optional dependencies.

### Preferred Commands (Pixi-first)

Use the tasks defined in [pixi.toml](pixi.toml) as the default workflow. Avoid raw `cmake` invocations unless a one-off advanced operation truly requires it.

```shell
# 1. Build the project (configures automatically)
pixi run build-tests

# 2. Run the full test suite
pixi run tests

# 3. Build the Debian/source package artifacts
pixi run build-pkg

# 4. Install the local build into /usr/local
pixi run install-local

# 5. Run the repo validation chain
pixi run ci-check
```

> **Important**: When switching between Pixi environments (e.g. from `tests` to `bench`), always clean `build/` first (`pixi run clean`) to avoid CMake cache collisions between different conda prefixes.

Direct shell calls such as `cmake -S . -B build ...` should remain exceptional, not the default. Use the Pixi task wrappers for configure, build, test, install, packaging, and docs generation whenever possible.

---

## 8. Skills Available in `.agents/skills/`

Antigravity provides specialized skills to assist with tinyopt development:
- **`tinyopt-dev-workflow`**: Step-by-step commands to configure, build, format, and debug tests.
- **`tinyopt-testing-and-validation`**: Test writing guide, derivative validation rules, Catch2 v3 templates, and edge case checklist.
- **`tinyopt-perf-and-architecture`**: Deep-dive into zero-allocation programming, Eigen expression templates, and optimization loop profiling.
