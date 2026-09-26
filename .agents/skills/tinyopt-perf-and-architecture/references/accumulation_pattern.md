# The Tinyopt Accumulation Pattern

The hallmark of Tinyopt's design is the **Direct Accumulation Pattern**. This document details why it makes Tinyopt fast and how to implement it correctly.

---

## 1. Traditional NLLS vs. Tinyopt Accumulation

### The Traditional Approach (e.g. Ceres Solver, g2o)
Traditional frameworks evaluate residual blocks individually, storing:
1. A monolithic residual vector: $r \in \mathbb{R}^M$
2. A monolithic sparse or dense Jacobian matrix: $J \in \mathbb{R}^{M \times N}$
3. A subsequent linear system assembly step:
   $$H = J^T J, \quad g = J^T r$$

For problems with thousands or millions of measurements (e.g. bundle adjustment, point cloud registration), storing $J$ consumes large amounts of RAM and frequently blows past CPU L1/L2/L3 caches.

### Tinyopt Direct Accumulation
Tinyopt allows problems to accumulate directly into the reduced linear system:

$$H = \sum_{k} J_k^T J_k \in \mathbb{R}^{N \times N}$$
$$g = \sum_{k} J_k^T r_k \in \mathbb{R}^{N}$$

Storage requirements are bounded by the dimension of parameter space $N \times N$ rather than the number of measurements $M$.

---

## 2. Implementing the Accumulation Pattern

When defining a residual function, take references to `grad` and `H`:

```cpp
auto residual_fn = [&](const auto &params, auto &grad, auto &H) {
  double total_error = 0.0;

  for (const auto &obs : observations) {
    // 1. Compute single observation residual and local Jacobian
    double r_k = obs.value - model(params, obs.x);
    Vec<N> J_k = compute_jacobian(params, obs.x);

    total_error += r_k * r_k;

    // 2. Direct accumulation without storing arrays of r_k or J_k
    if constexpr (!traits::is_nullptr_v<decltype(grad)>) {
      grad.noalias() -= J_k * r_k; // depending on sign convention
      if constexpr (!traits::is_nullptr_v<decltype(H)>) {
        H.noalias() += J_k * J_k.transpose();
      }
    }
  }

  return total_error;
};
```

---

## 3. Performance Benefits

1. **Cache Locality**: The accumulator matrix $H$ and vector $g$ easily fit inside CPU L1/L2 cache throughout the accumulation loop.
2. **Zero Allocation**: No heap vectors are resized or allocated per iteration.
3. **Parallelizable**: Splitting the observation loop across threads (e.g. OpenMP) and summing local accumulator matrices is trivial and lock-free.
