# Mathematical Gradient Verification in Tinyopt

In non-linear optimization, human calculation errors in Jacobians or Hessians are the #1 cause of slow convergence, divergence, or subtle numerical bugs.

Tinyopt provides built-in utilities in `<tinyopt/diff/gradient_check.h>` to mathematically verify derivatives before running any solver.

---

## 1. When to Use Gradient Checking

- **Always** in unit tests for new cost functions or residuals.
- **Before** running an optimizer on complex custom manifolds or loss functions.
- **When debugging** slow or failing convergence in test suites.

---

## 2. API Overview

### Checking Residuals Gradient
Used for Non-Linear Least Squares residuals where `residuals(x, grad, H)` evaluates $r(x)$ and updates $g += J^T r$ and $H += J^T J$:

```cpp
#include <tinyopt/diff/gradient_check.h>

// Function signature:
// CheckResidualsGradient(x0, residuals, eps = 1e-6, tol = 1e-4)
REQUIRE(diff::CheckResidualsGradient(x0, residuals));
```

### Checking Cost Gradient
Used for general unconstrained scalar objective functions $f(x)$ where `loss(x, grad, H)` evaluates $f(x)$ and computes $\nabla f(x)$:

```cpp
#include <tinyopt/diff/gradient_check.h>

// Function signature:
// CheckCostGradient(x0, cost_function, eps = 1e-6, tol = 1e-4)
REQUIRE(diff::CheckCostGradient(x0, loss));
```

---

## 3. How Gradient Check Works Under the Hood

The checker computes numerical derivatives using central finite differences:

$$\frac{\partial f}{\partial x_i} \approx \frac{f(x + \epsilon e_i) - f(x - \epsilon e_i)}{2\epsilon}$$

It then calculates the relative error between analytical gradient $g_{analytical}$ and numerical gradient $g_{numerical}$:

$$\text{Relative Error} = \frac{|g_{analytical} - g_{numerical}|}{\max(|g_{analytical}|, |g_{numerical}|, 1.0)}$$

If the relative error exceeds `tolerance` (default `1e-4` for float/double), the checker prints detailed component-wise mismatches to standard error and returns `false`.

---

## 4. Troubleshooting Gradient Mismatches

1. **Sign errors**: Check if you missed a negative sign in the chain rule (e.g. $\frac{\partial}{\partial x}(y - f(x)) = -f'(x)$).
2. **Missing accumulation terms**: For NLLS accumulation, ensure $J^T J$ accounts for all residual dimensions.
3. **Step size $\epsilon$ sensitivity**: For functions with severe scaling disparities or high second derivatives, adjust `eps`:
   ```cpp
   diff::CheckCostGradient(x0, loss, /*eps=*/1e-5, /*tol=*/1e-3);
   ```
4. **Non-differentiable points**: Avoid checking points directly on kinks (e.g. $|x| = 0$ in Huber loss transition).
