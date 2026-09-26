# Eigen Best Practices for High-Performance Optimization in Tinyopt

Eigen is a template-based linear algebra library that provides high-performance vectorization via expression templates. However, improper usage can introduce hidden memory allocations, unnecessary temporaries, or aliasing bugs.

Follow these rules across all Tinyopt components:

---

## 1. Zero Dynamic Allocation in Optimization Loops

- Use **Fixed-size Types** (`Vec2`, `Vec3`, `Mat33`, `Vector<T, N>`, `Matrix<T, Rows, Cols>`) whenever problem dimensionality is known at compile time.
  - Fixed-size types allocate on the stack, incur zero `malloc` overhead, and allow compilers to fully unroll loops and vectorize via AVX/NEON.
- If dynamic sizing is unavoidable (`VecX`, `MatX`):
  - Pre-allocate memory buffers before entering the optimization loop.
  - Never call `.resize()` inside cost evaluations or residual iterations.

---

## 2. Eliminate Temporary Matrices with `.noalias()`

When multiplying matrices in Eigen, the assignment operator conservatively introduces a temporary matrix to prevent aliasing bugs:

```cpp
// BAD: Allocates a temporary matrix before accumulating
H += J.transpose() * J;

// GOOD: Direct accumulation into H with zero temporaries
H.noalias() += J.transpose() * J;
g.noalias() += J.transpose() * r;
```

> **Warning**: Only use `.noalias()` when the target matrix on the LHS does NOT appear on the RHS of the expression.

---

## 3. Storage Order & Memory Alignment

- Eigen defaults to **Column-Major** storage.
- When iterating through matrix elements in nested loops, access column-by-column (outer loop over columns `j`, inner loop over rows `i`) for cache-line locality:
  ```cpp
  for (int j = 0; j < cols; ++j) {
    for (int i = 0; i < rows; ++i) {
      process(A(i, j));
    }
  }
  ```

---

## 4. Compile-time Dispatch with `if constexpr`

When evaluating residuals, solvers may request only the residual value, the residual + gradient, or the full Hessian. Use `if constexpr` to eliminate inactive branches at compile time:

```cpp
if constexpr (!traits::is_nullptr_v<decltype(grad)>) {
  grad.noalias() += J.transpose() * res;
  if constexpr (!traits::is_nullptr_v<decltype(H)>) {
    H.noalias() += J.transpose() * J;
  }
}
```
This guarantees zero runtime overhead when gradient or Hessian computation is unneeded.
