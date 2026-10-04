# IO
#option (USE_EIGEN  "Use Eigen library" ON) for now this must be ON
option(TINYOPT_USE_FMT "Use fmt formatting" OFF)

option(TINYOPT_ENABLE_FORMATTERS "Enable definion of std::formatter for streamable types, linked to TINYOPT_NO_FORMATTERS" ON)

# C API Options.
option(TINYOPT_BUILD_C_LIBRARY "Build the C ABI library" OFF)
set(TINYOPT_C_FIXED_SIZES "1;2;3;4;5;6;10;12" CACHE STRING
	"Fixed parameter dimensions generated for the C API")
set_property(CACHE TINYOPT_C_FIXED_SIZES PROPERTY STRINGS 1 2 3 4 5 6 10 12)
option(TINYOPT_C_API_FLOAT "Build the float C API" ON)
option(TINYOPT_BUILD_SHARED_C "Build tinyopt_c as a shared library (OFF builds static)" ON)

## Disable these to speed-up compilation if not needed
option(TINYOPT_DISABLE_AUTODIFF "Disable Automatic Differentiation in Optimizers" OFF)
option(TINYOPT_DISABLE_NUMDIFF "Disable Numeric Differentiation in Optimizers" OFF)
option(TINYOPT_ENFORCE_NO_DYNAMIC_ALLOCATIONS
	"Enforce no Eigen heap allocations for fixed-size optimizations" OFF)
option(TINYOPT_ENABLE_GAUSS_NEWTON "Enable the Gauss-Newton optimizer" ON)
option(TINYOPT_ENABLE_GRADIENT_DESCENT "Enable the Gradient Descent optimizer" OFF)
option(TINYOPT_ENABLE_CONJUGATE_GRADIENT "Enable the Conjugate Gradient optimizer" OFF)
option(TINYOPT_ENABLE_DOGLEG "Enable Powell's DogLeg optimizer" OFF)
option(TINYOPT_ENABLE_BFGS "Enable the BFGS optimizer" OFF)
option(TINYOPT_ENABLE_LBFGS "Enable the limited-memory BFGS optimizer" OFF)
option(TINYOPT_ENABLE_OPTIMIZERS_ALL "Enable all optimizers in global Optimize dispatch" OFF)
if(TINYOPT_ENABLE_OPTIMIZERS_ALL)
	set(TINYOPT_ENABLE_GAUSS_NEWTON ON)
	set(TINYOPT_ENABLE_GRADIENT_DESCENT ON)
	set(TINYOPT_ENABLE_CONJUGATE_GRADIENT ON)
	set(TINYOPT_ENABLE_DOGLEG ON)
	set(TINYOPT_ENABLE_BFGS ON)
	set(TINYOPT_ENABLE_LBFGS ON)
endif()

# Other decompositions are optional to keep default Eigen compile times low.
option(TINYOPT_ENABLE_LINEAR_SOLVER_LDLT "Enable dense and sparse LDLT solvers" ON)
option(TINYOPT_ENABLE_LINEAR_SOLVER_LLT "Enable dense and sparse LLT solvers" OFF)
option(TINYOPT_ENABLE_LINEAR_SOLVER_LU "Enable dense and sparse LU solvers" OFF)
option(TINYOPT_ENABLE_LINEAR_SOLVER_QR "Enable dense and sparse QR solvers" OFF)
option(TINYOPT_ENABLE_LINEAR_SOLVER_SVD "Enable the dense Jacobi SVD solver" OFF)
option(TINYOPT_ENABLE_LINEAR_SOLVER_ALL "Enable all built-in dense linear solvers" OFF)
if(TINYOPT_ENABLE_LINEAR_SOLVER_ALL)
	set(TINYOPT_ENABLE_LINEAR_SOLVER_LDLT ON)
	set(TINYOPT_ENABLE_LINEAR_SOLVER_LLT ON)
	set(TINYOPT_ENABLE_LINEAR_SOLVER_LU ON)
	set(TINYOPT_ENABLE_LINEAR_SOLVER_QR ON)
	set(TINYOPT_ENABLE_LINEAR_SOLVER_SVD ON)
endif()
option(TINYOPT_ENABLE_SUITESPARSE "Enable SuiteSparse CHOLMOD (review component/module licenses)" OFF)

# Treat compiler warnings as errors (disabled for pip builds with arbitrary compilers)
option(TINYOPT_WERROR "Build with -Werror" ON)

# Examples
option(TINYOPT_BUILD_EXAMPLES "Build examples" OFF) # Enable/Disable ALL examples

# Tests
option(TINYOPT_BUILD_TESTS "Build tests" ON) # Enable/Disable ALL tests
option(TINYOPT_BUILD_INSTALL_TESTS "Build isolated install smoke tests" OFF)
option(TINYOPT_BUILD_SOPHUS_TEST "Build Sophus tests" OFF)
option(TINYOPT_BUILD_LIEPLUSPLUS_TEST "Build Lie++ tests" OFF)

# Benchmarks
option(TINYOPT_BUILD_BENCHMARKS "Build benchmarks" OFF) # Enable/Disable ALL benchmarks
option(TINYOPT_BUILD_CERES "Build Ceres tests and benchmarks" OFF)
option(TINYOPT_BUILD_G2O_BENCHMARKS "Build g2o comparison benchmarks" OFF)
option(TINYOPT_BUILD_GTSAM_BENCHMARKS "Build GTSAM comparison benchmarks" OFF)

# Packages
option(TINYOPT_BUILD_PACKAGES "Build packages" OFF)
option(TINYOPT_BUILD_PIP_PACKAGE "Enable the pip install target" OFF)

# Documentation
option(TINYOPT_BUILD_DOCS "Build documentation" OFF)
