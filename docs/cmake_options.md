# CMake Options

Tinyopt's CMake options control optional algorithms, dependencies, build targets, and compile-time
features. Set an option when configuring the project with `-DNAME=VALUE`. For example:

```sh
cmake -S . -B build -G Ninja \
  -DTINYOPT_BUILD_TESTS=ON \
  -DTINYOPT_ENABLE_GRADIENT_DESCENT=ON \
  -DTINYOPT_ENABLE_LINEAR_SOLVER_QR=ON
```

The options and defaults below are defined in `cmake/Options.cmake`. Reconfigure after changing
options; compile-time features are exposed through the `tinyopt` interface target.

## Compile-Time Features

| Option | Default | Description |
| --- | --- | --- |
| `TINYOPT_USE_FMT` | `OFF` | Use the `fmt` formatting library when available. |
| `TINYOPT_ENABLE_FORMATTERS` | `ON` | Enable `std::formatter` specializations for streamable types. Disable to define `TINYOPT_NO_FORMATTERS`. |
| `TINYOPT_DISABLE_AUTODIFF` | `OFF` | Disable automatic differentiation in optimizers. Numerical differentiation can still be used unless separately disabled. |
| `TINYOPT_DISABLE_NUMDIFF` | `OFF` | Disable numerical differentiation in optimizers. |
| `TINYOPT_ENFORCE_NO_DYNAMIC_ALLOCATIONS` | `OFF` | For fixed-size optimization problems, prevent Eigen heap allocations in optimizer paths, retain only the first and latest history samples, omit the final Hessian, and disable optimizer logging. This cannot prevent allocations in user callbacks. |
| `TINYOPT_ENABLE_GAUSS_NEWTON` | `ON` | Enable the Gauss-Newton optimizer. |
| `TINYOPT_ENABLE_GRADIENT_DESCENT` | `OFF` | Enable the Gradient Descent optimizer. |
| `TINYOPT_ENABLE_CONJUGATE_GRADIENT` | `OFF` | Enable nonlinear conjugate gradient in global `Optimize()` dispatch. |
| `TINYOPT_ENABLE_DOGLEG` | `OFF` | Enable Powell's DogLeg method in global `Optimize()` dispatch. |

These optimizer switches only control selection through the global `Optimize()` function. The
solver and optimizer headers remain directly usable when a switch is `OFF`; unit tests for each
implementation are built independently of these switches. Enable the corresponding switch to use
`Options::Solver::ConjugateGradient` or `Options::Solver::DogLeg` with global `Optimize()`.

## Linear Solvers

Optional Eigen decompositions are disabled by default to reduce compile times. Enable only the
methods required by the application.

| Option | Default | Description |
| --- | --- | --- |
| `TINYOPT_ENABLE_LINEAR_SOLVER_LDLT` | `ON` | Enable dense and sparse LDLT. |
| `TINYOPT_ENABLE_LINEAR_SOLVER_LLT` | `OFF` | Enable dense and sparse LLT. |
| `TINYOPT_ENABLE_LINEAR_SOLVER_LU` | `OFF` | Enable dense and sparse LU. |
| `TINYOPT_ENABLE_LINEAR_SOLVER_QR` | `OFF` | Enable dense and sparse QR. |
| `TINYOPT_ENABLE_LINEAR_SOLVER_SVD` | `OFF` | Enable dense Jacobi SVD. |
| `TINYOPT_ENABLE_SUITESPARSE` | `OFF` | Enable SuiteSparse CHOLMOD. Review the licenses of the selected SuiteSparse components. |

## Build Targets

| Option | Default | Description |
| --- | --- | --- |
| `TINYOPT_BUILD_EXAMPLES` | `OFF` | Build examples. |
| `TINYOPT_BUILD_TESTS` | `ON` | Build the test suite. |
| `TINYOPT_BUILD_INSTALL_TESTS` | `OFF` | Add isolated install smoke tests. |
| `TINYOPT_BUILD_SOPHUS_TEST` | `OFF` | Build tests that depend on Sophus. |
| `TINYOPT_BUILD_LIEPLUSPLUS_TEST` | `OFF` | Build tests that depend on Lie++. |
| `TINYOPT_BUILD_BENCHMARKS` | `OFF` | Build benchmark targets. |
| `TINYOPT_BUILD_CERES` | `OFF` | Build Ceres comparison tests and benchmarks. |
| `TINYOPT_BUILD_PACKAGES` | `OFF` | Enable package targets. |
| `TINYOPT_BUILD_PIP_PACKAGE` | `OFF` | Enable the `pip-install` target. |
| `TINYOPT_BUILD_DOCS` | `OFF` | Enable the Sphinx and Doxygen documentation targets. Requires the documentation tools. |