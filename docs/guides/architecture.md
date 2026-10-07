# Tinyopt Architectural Design & System Overview

Tinyopt is a header-only C++20 numerical optimization library designed from the ground up for high computational efficiency, low latency, and zero dynamic memory allocations in critical paths.

---

## 1. System Layering

Tinyopt is organized into clear, decoupled architectural layers:

```
┌─────────────────────────────────────────────────────────────────┐
│                    1. High-Level User API                       │
│             Optimize(x, cost_or_residuals, options)            │
│               tinyopt/optimize.h, tinyopt/tinyopt.h             │
└────────────────────────────────┬────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────┐
│                 2. Optimizers (Optimizer1/2)                    │
│       Iteration loop, algorithm steps, trust region, damping    │
│     tinyopt/optimizers/optimizer.h, optimizer1.h, optimizer2.h  │
└────────────────────────────────┬────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────┐
│                  3. Linear Algebra Backends                     │
│        Linear system decomposition and step solution             │
│              tinyopt/math/linear_solvers.h                      │
└────────────────────────────────┬────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────┐
│                 4. Differentiation & Losses                     │
│    AutoDiff (Jets), NumDiff, Robust M-Estimators (Huber, Tukey) │
│               tinyopt/diff/, tinyopt/losses/                    │
└────────────────────────────────┬────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────┐
│               5. Core Types & Metaprogramming                   │
│     Compile-time traits, Eigen wrappers, manifold interfaces    │
│           tinyopt/types.h, tinyopt/traits.h, 3rdparty/          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Core Architectural Philosophy: The Accumulation Pattern

### Traditional Least Squares (Monolithic Buffering)
Traditional non-linear least squares solvers (such as Ceres Solver or g2o) evaluate residuals by allocating arrays for:
1. All residuals: $r = [r_1, r_2, \dots, r_M]^T \in \mathbb{R}^M$
2. Full Jacobian matrix: $J \in \mathbb{R}^{M \times N}$
3. Assembling the normal equations via $H = J^T J$ and $g = J^T r$

When $M$ (number of observations) is large (e.g. 100,000 points in vision/robotics), storing $J$ requires massive dynamic memory allocations, triggering cache misses and memory bandwidth saturation.

### Tinyopt Direct Accumulation
Tinyopt solves this through its **Accumulation Pattern**:
Instead of storing $r$ and $J$, users and cost evaluators accumulate directly into the linear system:

$$H = \sum_{i=1}^M J_i^T J_i \in \mathbb{R}^{N \times N}$$
$$g = \sum_{i=1}^M J_i^T r_i \in \mathbb{R}^N$$

- **Memory Bound**: Memory usage is bounded by parameter size $N \times N$, completely independent of measurement count $M$.
- **Cache Locality**: $H$ and $g$ reside continuously in CPU L1/L2 cache throughout the accumulation pass.
- **Zero Allocations**: Fixed-size solver state uses stack-backed Eigen types during iterations.

For deployments that require Tinyopt itself to avoid dynamic allocation on fixed-size problems,
configure with `-DTINYOPT_ENFORCE_NO_DYNAMIC_ALLOCATIONS=ON`. This uses five-entry FIFO output
histories, omits the final Hessian, disables optimizer logging, and activates Eigen's runtime
no-malloc guard. It does not apply to dynamic-size parameters or prevent heap allocations made by
user residuals, accumulation functions, or callbacks.

---

## 3. Differentiation Architecture

Tinyopt provides three complementary approaches to differentiation:

### 1. Dual Numbers / Jets (Automatic Differentiation)
- Implemented in `tinyopt/diff/jet.h`.
- Implements forward-mode automatic differentiation using dual numbers:
  $$f(a + b\epsilon) = f(a) + f'(a)b\epsilon, \quad \text{where } \epsilon^2 = 0$$
- Allows users to write templated functions (`auto loss = [](const auto &x) { ... }`) and obtain exact machine-precision derivatives without manual differentiation.

### 2. Analytical Differentiation
- The fastest method available. Users write an accumulation lambda taking `(const auto &x, auto &grad, auto &H)`.
- Derivatives are populated directly via Eigen vector/matrix operations.

### 3. Numerical Differentiation (Finite Differences)
- Implemented in `tinyopt/diff/num_diff.h`.
- Uses central differences:
  $$\frac{\partial f}{\partial x_i} \approx \frac{f(x + \epsilon e_i) - f(x - \epsilon e_i)}{2\epsilon}$$
- Primarily utilized as an automated oracle in `tinyopt/diff/gradient_check.h` to mathematically verify analytical Jacobians in unit tests.

---

## 4. Solvers & Damping Logic

### Levenberg-Marquardt (LM)
Tinyopt's LM optimizer solves regularized normal equations:

$$(H + \lambda D) \delta x = -g$$

Where:
- $H \approx J^T J$ is the Gauss-Newton Hessian approximation.
- $D$ is either the identity matrix $I$ or $\text{diag}(H)$.
- $\lambda$ is dynamically adapted using the gain ratio $\rho$:
  $$\rho = \frac{\text{Actual Cost Reduction}}{\text{Predicted Model Reduction}}$$
  - If $\rho > 0$, the step is accepted and $\lambda$ is decreased.
  - If $\rho \le 0$, the step is rejected, $\lambda$ is increased, and the linear system is re-solved.

### Gauss-Newton (GN)
Direct solution without damping ($\lambda = 0$). Ideal for well-conditioned, near-linear problems near the basin of attraction.

### Gradient Descent (GD)
First-order steepest descent step: $\delta x = -\alpha g$, with line search or adaptive step sizing.

---

## 5. Manifold & Lie Group Integration

For parameters living on non-Euclidean manifolds (e.g. 3D rotations in $SO(3)$ or rigid poses in $SE(3)$):
- Parameter updates use the exponential map (retraction / plus operator):
  $$x \leftarrow x \boxplus \delta x = x \circ \exp(\delta x)$$
- Adapters in `include/tinyopt/3rdparty/` support [Sophus](https://github.com/strasdat/Sophus) and `Lie++`.
