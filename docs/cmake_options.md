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
| `TINYOPT_ENABLE_BFGS` | `OFF` | Enable full-memory BFGS in global `Optimize()` dispatch. |
| `TINYOPT_ENABLE_LBFGS` | `OFF` | Enable limited-memory BFGS in global `Optimize()` dispatch. |
| `TINYOPT_ENABLE_OPTIMIZERS_ALL` | `OFF` | Enable Gauss-Newton, Gradient Descent, Conjugate Gradient, DogLeg, BFGS, and L-BFGS in global `Optimize()` dispatch. |

The `TINYOPT_ENABLE_GAUSS_NEWTON`, `TINYOPT_ENABLE_GRADIENT_DESCENT`,
`TINYOPT_ENABLE_CONJUGATE_GRADIENT`, `TINYOPT_ENABLE_DOGLEG`, `TINYOPT_ENABLE_BFGS`, and
`TINYOPT_ENABLE_LBFGS` switches control availability through the global `Optimize()` dispatch only.
They do not prevent direct use: include an optimizer's header and instantiate its optimizer class.
For example, Conjugate Gradient can be used directly even when
`TINYOPT_ENABLE_CONJUGATE_GRADIENT` is `OFF`:

```cpp
#include <tinyopt/optimizers/cg.h>

Eigen::VectorXd x = Eigen::VectorXd::Ones(10);
auto cost = [](const auto &value) { return value.squaredNorm(); };
tinyopt::cg::Optimizer<Eigen::VectorXd> optimizer;
auto sum = optimizer.Optimize(x, cost);
```

Set `TINYOPT_ENABLE_CONJUGATE_GRADIENT=ON` to select it through global `Optimize()` using
`Options::Solver::ConjugateGradient`. The same distinction applies to the other optimizer switches.
Unit tests for each implementation are built independently of these switches.

Use `-DTINYOPT_ENABLE_OPTIMIZERS_ALL=ON` to enable every optimizer in one step. SuiteSparse and
linear solver backends are controlled by their separate options.

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
| `TINYOPT_ENABLE_LINEAR_SOLVER_ALL` | `OFF` | Enable all built-in dense solvers: LDLT, LLT, LU, QR, and SVD. SuiteSparse remains controlled separately. |
| `TINYOPT_ENABLE_SUITESPARSE` | `OFF` | Enable SuiteSparse CHOLMOD. Review the licenses of the selected SuiteSparse components. With `TINYOPT_BUILD_C_LIBRARY`, also builds the sparse C API (`c_api_sparse.h`). |

## Build Targets

| Option | Default | Description |
| --- | --- | --- |
| `TINYOPT_C_FIXED_SIZES` | `1;2;3;4;5;6;10;12` | Fixed parameter dimensions to generate for the C API. Each enabled dimension generates double (`_d`) APIs, and float (`_f`) APIs when `TINYOPT_C_API_FLOAT` is enabled. |
| `TINYOPT_C_API_FLOAT` | `ON` | Build and expose the float C API, including its dynamic and generated fixed-size entrypoints. |
| `TINYOPT_BUILD_C_LIBRARY` | `OFF` | Build the optional `tinyopt_c` C ABI library and its C tests. |
| `TINYOPT_BUILD_SHARED_C` | `ON` | Build `tinyopt_c` as a shared library. Set `OFF` to build a static C library instead. |
| `TINYOPT_BUILD_WASM` | `OFF` | Build the WebAssembly module and its JavaScript example. Requires Emscripten (`emcmake`) and `TINYOPT_BUILD_C_LIBRARY=ON`; see `pixi run build-wasm`. |
| `TINYOPT_BUILD_EXAMPLES` | `OFF` | Build examples. |
| `TINYOPT_BUILD_TESTS` | `ON` | Build the test suite. |
| `TINYOPT_BUILD_INSTALL_TESTS` | `OFF` | Add isolated install smoke tests. |
| `TINYOPT_BUILD_SOPHUS_TEST` | `OFF` | Build tests that depend on Sophus. |
| `TINYOPT_BUILD_LIEPLUSPLUS_TEST` | `OFF` | Build tests that depend on Lie++. |
| `TINYOPT_BUILD_BENCHMARKS` | `OFF` | Build benchmark targets. |
| `TINYOPT_BUILD_CERES` | `OFF` | Build Ceres comparison tests and benchmarks. |
| `TINYOPT_BUILD_G2O_BENCHMARKS` | `OFF` | Build g2o comparison benchmarks. |
| `TINYOPT_BUILD_GTSAM_BENCHMARKS` | `OFF` | Build GTSAM comparison benchmarks. |
| `TINYOPT_BUILD_PACKAGES` | `OFF` | Enable package targets. |
| `TINYOPT_BUILD_PIP_PACKAGE` | `OFF` | Enable the `pip-install` target. |
| `TINYOPT_BUILD_DOCS` | `OFF` | Enable the Sphinx and Doxygen documentation targets. Requires the documentation tools. |