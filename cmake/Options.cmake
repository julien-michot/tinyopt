# IO
#option (USE_EIGEN  "Use Eigen library" ON) for now this must be ON
option(TINYOPT_USE_FMT "Use fmt formatting" OFF)

option(TINYOPT_ENABLE_FORMATTERS "Enable definion of std::formatter for streamable types, linked to TINYOPT_NO_FORMATTERS" ON)

# Build Options.
## Disable these to speed-up compilation if not needed
option(TINYOPT_DISABLE_AUTODIFF "Disable Automatic Differentiation in Optimizers" OFF)
option(TINYOPT_DISABLE_NUMDIFF "Disable Numeric Differentiation in Optimizers" OFF)
option(TINYOPT_ENFORCE_NO_DYNAMIC_ALLOCATIONS
	"Enforce no Eigen heap allocations for fixed-size optimizations" OFF)

# Other decompositions are optional to keep default Eigen compile times low.
option(TINYOPT_ENABLE_LINEAR_SOLVER_LDLT "Enable dense and sparse LDLT solvers" ON)
option(TINYOPT_ENABLE_LINEAR_SOLVER_LLT "Enable dense and sparse LLT solvers" OFF)
option(TINYOPT_ENABLE_LINEAR_SOLVER_LU "Enable dense and sparse LU solvers" OFF)
option(TINYOPT_ENABLE_LINEAR_SOLVER_QR "Enable dense and sparse QR solvers" OFF)
option(TINYOPT_ENABLE_LINEAR_SOLVER_SVD "Enable the dense Jacobi SVD solver" OFF)

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

# Packages
option(TINYOPT_BUILD_PACKAGES "Build packages" OFF)
option(TINYOPT_BUILD_PIP_PACKAGE "Enable the pip install target" OFF)

# Documentation
option(TINYOPT_BUILD_DOCS "Build documentation" OFF)
