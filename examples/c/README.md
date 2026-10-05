# Tinyopt C Examples

The examples use the C API described in the [C API guide](../../docs/api/c.md). They link against
the library built by `pixi run build-c-library` (`build-c-library/`), for instance:

```sh
pixi run build-c-library
cc examples/c/quadratic.c -Iinclude -Ibuild-c-library/generated/include \
   -Lbuild-c-library -ltinyopt_c -Wl,-rpath,"$PWD/build-c-library" -o quadratic && ./quadratic
```

With `-DTINYOPT_BUILD_EXAMPLES=ON` CMake also builds them (as `tinyopt_example_c_*`).

| Example | Evaluation mode | Problem |
| --- | --- | --- |
| `quadratic.c` | `TINYOPT_EVAL_RESIDUALS` | Smallest dynamic-size fit, with a hand-written Jacobian. |
| `fixed_circle_fit.c` | `TINYOPT_EVAL_RESIDUALS` | Circle fit with fixed-size 3 parameters (`tinyopt_optimize3`). |
| `dynamic_cost.c` | `TINYOPT_EVAL_COST_ONLY` | Scalar objective only, the gradient is estimated numerically. |
| `manual_gradient.c` | `TINYOPT_EVAL_GRADIENT` | Gradient descent on a hand-accumulated cost and gradient. |
| `manual_hessian.c` | `TINYOPT_EVAL_HESSIAN` | Hand-accumulated cost, gradient and Hessian. |
| `sparse_hessian.c` | sparse Hessian | Tridiagonal signal smoothing with CHOLMOD (needs `TINYOPT_ENABLE_SUITESPARSE=ON`). |
