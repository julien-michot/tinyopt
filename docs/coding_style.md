# Tinyopt Coding Style Guide

Tinyopt is an ultra-high-performance, header-only C++20 library. Maintaining code uniformity, readability, and hardware efficiency is critical across the codebase.

All contributions must strictly adhere to the standards outlined in this guide.

---

## 1. Automated Formatting (`.clang-format`)

Tinyopt enforces formatting via Clang-Format using a Google-based standard with the following key rules:
- **Indentation**: 2 spaces (no tabs).
- **Line Limit**: 100 characters.
- **Braces**: Attached style (`Attach`), no line break before opening braces for classes, functions, or control statements.
- **Access Modifiers**: Indented by -1 relative to class (`public:`, `private:`).
- **Pointer & Reference Alignment**: Left/Derive (`const Vec3 &x`, `double *grad`).

### Running Clang-Format
Before submitting or staging changes, run the repository formatter:
```shell
./scripts/format.sh
```
Or check for violations without modifying files:
```shell
./scripts/format.sh --check
```

---

## 2. File & License Headers

Every `.h`, `.hpp`, and `.cpp` file must begin with the official Apache-2.0 copyright header:

```cpp
// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0
```

Headers must use `#pragma once` as the include guard.

---

## 3. Include Ordering

Includes must be grouped with blank lines separating each block:
1. Related main header (for `.cpp` files)
2. Standard C/C++ library headers (`<cmath>`, `<vector>`, `<type_traits>`, `<concepts>`)
3. Third-party library headers (`<Eigen/Dense>`, `<catch2/...>`, `<sophus/...>`)
4. Tinyopt library headers (`<tinyopt/types.h>`, `<tinyopt/diff/jet.h>`)

Example:
```cpp
// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>
#include <concepts>
#include <type_traits>

#include <Eigen/Core>

#include <tinyopt/traits.h>
#include <tinyopt/types.h>
```

---

## 4. Naming Conventions

| Entity | Convention | Example |
| :--- | :--- | :--- |
| **Types, Classes, Structs** | `PascalCase` | `LevenbergMarquardt`, `Jet`, `CostFunction` |
| **Type Aliases / Templates** | `PascalCase` | `Vec3`, `Mat33`, `Scalar`, `Derived` |
| **Functions & Methods** | `PascalCase` (public API) / `camelCase` (helpers) | `Optimize(...)`, `CheckResidualsGradient(...)`, `coord(...)` |
| **Variables & Members** | `snake_case` | `max_iters`, `damping_lambda`, `step_norm` |
| **Private Member Variables** | `snake_case_` (trailing underscore) | `options_`, `workspace_`, `current_cost_` |
| **Constants & Enum Values** | `PascalCase` | `StopReason::MaxItersReached`, `StopReason::GradientTolerance` |
| **Namespaces** | `snake_case` (lowercase) | `tinyopt`, `tinyopt::diff`, `tinyopt::solvers` |
| **Concepts** | `PascalCase` | `VectorSpace`, `ManifoldParameter` |

---

## 5. Modern C++20 Idioms & Best Practices

### Const-Correctness
- Mark all non-mutating member functions `const`.
- Mark non-mutating variables and parameters `const`.
- Pass cheap-to-copy types (scalar values, primitive types $\le 16$ bytes) by value. Pass larger structures, matrices, and parameters by `const &`.

### Compile-Time Optimization (`constexpr` & `if constexpr`)
- Mark pure utility functions and traits `constexpr`.
- Use `if constexpr` to eliminate dead code paths at compile time:
  ```cpp
  if constexpr (!traits::is_nullptr_v<decltype(grad)>) {
    grad.noalias() += J.transpose() * res;
  }
  ```

### `[[nodiscard]]` Attribute
Use `[[nodiscard]]` on all non-mutating functions that compute or return a value (getters, math functions, status checkers, factory methods):
```cpp
[[nodiscard]] inline bool Converged() const noexcept {
  return stop_reason == StopReason::GradientTolerance ||
         stop_reason == StopReason::StepTolerance;
}
```

### No Dynamic Allocations in Tight Loops
- **Never** call `malloc`, `new`, `std::vector::resize`, or create dynamic Eigen matrices (`MatrixXd`, `VectorXd`) inside optimization iterations or residual callbacks.
- Use fixed-size Eigen types whenever dimensions are known at compile-time (`Vec2`, `Vec3`, `Mat33`, `Matrix<T, Rows, Cols>`).

### Eigen `.noalias()` Discipline
When assigning matrix-matrix or matrix-vector products, always use `.noalias()` if the destination does not appear on the right-hand side:
```cpp
// Correct: Zero temporaries
H.noalias() += J.transpose() * J;
g.noalias() += J.transpose() * r;

// Avoid: Allocates hidden heap/stack temporary
H += J.transpose() * J;
```
