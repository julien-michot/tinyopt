# IO
#option (USE_EIGEN  "Use Eigen library" ON) for now this must be ON
option(TINYOPT_USE_FMT "Use fmt formatting" OFF)

option(TINYOPT_ENABLE_FORMATTERS "Enable definion of std::formatter for streamable types, linked to TINYOPT_NO_FORMATTERS" ON)

# Build Options.
## Disable these to speed-up compilation if not needed
option(TINYOPT_DISABLE_AUTODIFF "Disable Automatic Differentiation in Optimizers" OFF)
option(TINYOPT_DISABLE_NUMDIFF "Disable Numeric Differentiation in Optimizers" OFF)

# Examples
option(TINYOPT_BUILD_EXAMPLES "Build examples" OFF)
option(TINYOPT_BUILD_SOPHUS_EXAMPLES "Build Sophus examples, fecth it if not found" OFF)
option(TINYOPT_BUILD_LIEPLUSPLUS_EXAMPLES "Build Lie++ examples, fecth it if not found" OFF)

# Tests
option(TINYOPT_BUILD_TESTS "Build tests" ON)

# Benchmarks
option(TINYOPT_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(TINYOPT_BUILD_CERES "Build Ceres benchmarks" OFF)

# Packages
option(TINYOPT_BUILD_PACKAGES "Build packages" OFF)

# Documentation
option(TINYOPT_BUILD_DOCS "Build documentation" OFF)


# Adding Definitions
if (NOT TINYOPT_ENABLE_FORMATTERS)
  add_definitions(-DTINYOPT_NO_FORMATTERS=1)
endif ()
if (TINYOPT_DISABLE_AUTODIFF)
  add_definitions(-DTINYOPT_DISABLE_AUTODIFF=1)
endif ()
if (TINYOPT_DISABLE_NUMDIFF)
  add_definitions(-DTINYOPT_DISABLE_NUMDIFF=1)
endif ()
