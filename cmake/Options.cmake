# IO
#option (USE_EIGEN  "Use Eigen library" ON) for now this must be ON
option(TINYOPT_USE_FMT "Use fmt formatting" OFF)

option(TINYOPT_ENABLE_FORMATTERS "Enable definion of std::formatter for streamable types, linked to TINYOPT_NO_FORMATTERS" ON)

# Build Options.
## Disable these to speed-up compilation if not needed
option(TINYOPT_DISABLE_AUTODIFF "Disable Automatic Differentiation in Optimizers" OFF)
option(TINYOPT_DISABLE_NUMDIFF "Disable Numeric Differentiation in Optimizers" OFF)

# Examples
option(TINYOPT_BUILD_EXAMPLES "Build examples" OFF) # Enable/Disable ALL examples

# Tests
option(TINYOPT_BUILD_TESTS "Build tests" ON) # Enable/Disable ALL tests
option(TINYOPT_BUILD_SOPHUS_TEST "Build Sophus tests" OFF)
option(TINYOPT_BUILD_LIEPLUSPLUS_TEST "Build Lie++ tests" OFF)

# Benchmarks
option(TINYOPT_BUILD_BENCHMARKS "Build benchmarks" OFF) # Enable/Disable ALL benchmarks
option(TINYOPT_BUILD_CERES "Build Ceres tests and benchmarks" OFF)

# Packages
option(TINYOPT_BUILD_PACKAGES "Build packages" OFF)

# Documentation
option(TINYOPT_BUILD_DOCS "Build documentation" OFF)


# Linear Solvers
option(TINYOPT_BUILD_SOLVER_LDLT "Build Cholesky/LDLT linear solver" ON)
option(TINYOPT_BUILD_SOLVER_QR "Build (col-piv) QR linear solver" ON)
option(TINYOPT_BUILD_SOLVER_LU "Build (partial) LU linear solver" ON)
option(TINYOPT_BUILD_SOLVER_SVD "Build SVD linear solver" ON)

# Adding Definitions # TODO: use target_compile_definitions()
if (NOT TINYOPT_ENABLE_FORMATTERS)
  add_definitions(-DTINYOPT_NO_FORMATTERS=1)
endif ()
if (TINYOPT_DISABLE_AUTODIFF)
  add_definitions(-DTINYOPT_DISABLE_AUTODIFF=1)
endif ()
if (TINYOPT_DISABLE_NUMDIFF)
  add_definitions(-DTINYOPT_DISABLE_NUMDIFF=1)
endif ()
# Linear solvers
if (TINYOPT_BUILD_SOLVER_LDLT)
  add_definitions(-DTINYOPT_BUILD_SOLVER_LDLT=1)
endif ()
if (TINYOPT_BUILD_SOLVER_QR)
  add_definitions(-DTINYOPT_BUILD_SOLVER_QR=1)
endif ()
if (TINYOPT_BUILD_SOLVER_LU)
  add_definitions(-DTINYOPT_BUILD_SOLVER_LU=1)
endif ()
if (TINYOPT_BUILD_SOLVER_SVD)
  add_definitions(-DTINYOPT_BUILD_SOLVER_SVD=1)
endif ()
