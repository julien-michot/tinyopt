---
name: tinyopt-testing-and-validation
description: >-
  Use this skill when authoring new tests, debugging failing tests, validating mathematical
  gradients and Hessians, verifying optimizer convergence, or benchmarking algorithms in Tinyopt.
---

# Tinyopt Testing and Validation

This skill defines the rigorous standards and procedures required for authoring and validating tests in Tinyopt.

Before refactoring or running code experiments, read [the repository refactoring and experiment guidelines](../../../docs/development/refactoring-and-experiments.md).
For code changes under `include/` or `src/`, run `pixi run refactor-timings` as the final validation step after the correctness checks.

---

## 1. Golden Rules for Tinyopt Tests

1. **Verify Mathematical Derivatives First**:
   Never trust an analytical Jacobian/Hessian calculation without checking it against numerical differentiation. Always invoke `diff::CheckResidualsGradient` or `diff::CheckCostGradient`.
   See [gradient_checking.md](./references/gradient_checking.md) for details.

2. **Test Both Analytical & Automatic Differentiation (Jets)**:
   Whenever possible, write tests that exercise both manual derivative accumulation and Jet-based autodiff.

3. **Check Optimality & Convergence**:
   - `REQUIRE(out.Succeeded());`
   - `REQUIRE(out.Converged());`
   - `REQUIRE(std::abs(x - ground_truth) == Approx(...).margin(...));`
   - Check gradient norm reduction: final $||\nabla f(x^*)||$ should be sufficiently small.

4. **Multi-condition Coverage with Catch2 Generators**:
   Use `GENERATE(...)` to test multiple initial conditions (including points close to the minimum and points far away in the non-linear valley).

---

## 2. Test Authoring Procedure

### Step 1: Write Test Implementation
Create your test file in `tests/<feature_name>.cpp`. You can adapt the reference implementation in [test_template.cpp](./examples/test_template.cpp).

### Step 2: Register in `tests/CMakeLists.txt`
Add the executable and test target:
```cmake
add_executable(tinyopt_test_<feature_name> <feature_name>.cpp)
add_test_target(tinyopt_test_<feature_name>)
```

### Step 3: Compile and Execute
```shell
./.agents/skills/tinyopt-dev-workflow/scripts/run_tests.sh tinyopt_test_<feature_name>
```

---

## 3. Edge Case Checklist

Before concluding test development, verify how your code handles:
- [ ] Ill-conditioned initial guesses (saddle points, near-singular Hessian).
- [ ] Residuals that evaluate to zero or near machine epsilon.
- [ ] Iteration limits (`options.stop.max_iters`) triggering `StopReason::MaxItersReached`.
- [ ] NaN / Inf propagation and solver recovery or termination.
- [ ] Damping parameter updates ($\lambda$) under consecutive step failures.
