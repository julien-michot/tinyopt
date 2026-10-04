# Installation and Usage

Tinyopt's C++ API is header-only. The optional C ABI is built as `tinyopt_c`; it installs the C
headers and shared library alongside the C++ headers. Choose the API in the application source by
including either `<tinyopt/tinyopt.h>` or `<tinyopt/c/c_api.h>`.

## Install

Build and install the default C++ package:

```sh
cmake -S . -B build -G Ninja -DTINYOPT_BUILD_C_LIBRARY=OFF
cmake --build build
cmake --install build --prefix "$HOME/.local"
```

To include the C ABI and generated fixed-size C entrypoints, configure with
`-DTINYOPT_BUILD_C_LIBRARY=ON`:

```sh
cmake -S . -B build-c -G Ninja -DTINYOPT_BUILD_C_LIBRARY=ON
cmake --build build-c --target tinyopt_c
cmake --install build-c --prefix "$HOME/.local"
```

The C library is shared by default. `TINYOPT_C_API_FLOAT` controls float entrypoints, and
`TINYOPT_C_FIXED_SIZES` controls the generated dimensions. The generated per-dimension headers and
the precision aggregators `c_api_fixed_double.h` and `c_api_fixed_float.h` are installed under
`include/tinyopt/c`. SuiteSparse support is separate and remains disabled unless
`-DTINYOPT_ENABLE_SUITESPARSE=ON` is requested. Package builds can be produced with
`pixi run build-pkg` on Linux.

## C++ Consumer

Use the installed CMake package and link the `tinyopt` interface target. It supplies the include
directory, C++20 requirement, Eigen dependency, and the compile-time features selected when Tinyopt
was built:

```cmake
cmake_minimum_required(VERSION 3.25)
project(cpp_example LANGUAGES CXX)

find_package(Tinyopt CONFIG REQUIRED)
add_executable(cpp_example main.cpp)
target_link_libraries(cpp_example PRIVATE tinyopt)
```

`main.cpp` can include `<tinyopt/tinyopt.h>`. For a non-standard install prefix, set
`CMAKE_PREFIX_PATH` to that prefix when configuring the consumer:

```sh
cmake -S . -B build -DCMAKE_PREFIX_PATH="$HOME/.local"
```

## C Consumer

The Python package (`python -m pip install .`) bundles the C library, see [Python API](python.md).

The C ABI must have been enabled when Tinyopt was built. After `find_package`, locate the installed
`tinyopt_c` library and link it to the C executable:

```cmake
cmake_minimum_required(VERSION 3.25)
project(c_example LANGUAGES C)

find_package(Tinyopt CONFIG REQUIRED)
find_library(TINYOPT_C_LIBRARY NAMES tinyopt_c
  HINTS "${Tinyopt_INCLUDE_DIRS}/../lib"
  REQUIRED)

add_executable(c_example main.c)
target_include_directories(c_example PRIVATE "${Tinyopt_INCLUDE_DIRS}")
target_link_libraries(c_example PRIVATE "${TINYOPT_C_LIBRARY}")
```

For example, `main.c` can minimize a one-dimensional residual:

```c
#include <math.h>
#include <tinyopt/c/c_api.h>

static int residual(const double *x, int dims, double *out, double **jacobian,
                    int num_residuals, void *user_data) {
  (void)user_data;
  if (dims != 1 || num_residuals != 1) return 1;
  out[0] = x[0] - 2.0;
  if (*jacobian != NULL) (*jacobian)[0] = 1.0;
  return 0;
}

int main(void) {
  double x[] = {0.0};
  tinyopt_params_t params = {x, 1, NULL};
  tinyopt_problem_t problem = {0};
  problem.type = TINYOPT_EVAL_RESIDUALS;
  problem.fn.residuals = residual;
  problem.num_residuals = 1;
  if (tinyopt_optimize(&params, &problem, NULL, NULL) != TINYOPT_STATUS_OK) return 1;
  return fabs(x[0] - 2.0) > 1e-5;
}
```

On Linux, a standard install prefix such as `/usr` or `/usr/local` is already on the runtime
library search path. For a private prefix, add its `lib` directory to `LD_LIBRARY_PATH` when running
the executable.