---
name: tinyopt-perf-and-architecture
description: >-
  Use this skill when designing high-performance optimization algorithms, eliminating memory
  allocations in inner loops, utilizing Eigen expression templates (.noalias()), implementing
  Tinyopt's direct accumulation pattern, or benchmarking against Ceres.
---

# Tinyopt Performance & Architecture

This skill provides architectural guidelines for developing zero-allocation, SIMD-vectorized optimization routines in Tinyopt.

---

## 1. Architectural Core Principles

Tinyopt is engineered for maximum throughput on mathematical optimization workloads:
1. **Direct Accumulation**: Accumulate $J^T J$ and $J^T r$ directly instead of assembling massive Jacobian blocks in RAM.
   See [accumulation_pattern.md](./references/accumulation_pattern.md).
2. **Zero Heap Allocations**: All inner-loop structures must use stack-allocated fixed-size Eigen matrices (`Vec2`, `Vec3`, `Mat33`, `Vector<T, N>`) or pre-allocated scratch buffers.
3. **Expression Template Discipline**: Always append `.noalias()` when assigning products to avoid hidden heap copies.
   See [eigen_best_practices.md](./references/eigen_best_practices.md).
4. **Compile-Time Dispatch**: Use `if constexpr` and type traits (`traits::is_nullptr_v`) to eliminate unused branches at compile-time.

---

## 2. Optimizing an Inner Loop: Quick Checklist

Before finalizing any performance-sensitive algorithm or cost function:
- [ ] Are all inner-loop variables stack-allocated (fixed-size Eigen types)?
- [ ] Are matrix product assignments marked with `.noalias()`?
- [ ] Is memory accessed in column-major order?
- [ ] Are function arguments passed by `const &` (or by value for small types <= 16 bytes)?
- [ ] Are all unused gradient/Hessian computations guarded by `if constexpr (!traits::is_nullptr_v<...>)`?

---

## 3. Benchmarking Against Ceres Solver

To measure speed and convergence iterations relative to Ceres Solver:
```shell
# 1. Clean and configure in benchmark environment
pixi run clean

# 2. Build benchmark suite
pixi run build-bench

# 3. Execute benchmarks
pixi run bench
```
Ensure your changes do not introduce latency regressions or increase the memory footprint.
