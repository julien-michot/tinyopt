# IO
#option (USE_EIGEN  "Use Eigen library" ON) for now this must be ON
option(USE_FMT "Use fmt formatting" OFF)

option(ENABLE_FORMATTERS "Enable definion of std::formatter for streamable types, linked to TINYOPT_NO_FORMATTERS" ON)

# Build Options
# Examples
option(BUILD_TINYOPT_EXAMPLES "Build examples" OFF)
# Tests
option(BUILD_TINYOPT_TESTS "Build tests" ON)
option(BUILD_TINYOPT_SOPHUS_EXAMPLES "Build Sophus examples, fecth it if not found" OFF)
option(BUILD_TINYOPT_LIEPLUSPLUS_EXAMPLES "Build Lie++ examples, fecth it if not found" OFF)
# Packages
option(BUILD_TINYOPT_PACKAGES "Build packages" OFF)
# Documentation
option(BUILD_TINYOPT_DOCS "Build documentation" OFF)


# Adding Definitions
if (NOT ENABLE_FORMATTERS)
  add_definitions(-DTINYOPT_NO_FORMATTERS=1)
endif ()
