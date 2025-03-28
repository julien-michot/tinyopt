# IO
#option (USE_EIGEN  "Use Eigen library" ON) for now this must be ON
option(USE_FMT "Use fmt formatting" OFF)

option(ENABLE_FORMATTERS "Enable definion of std::formatter for streamable types, linked to TINYOPT_NO_FORMATTERS" ON)

# Build Options
#option (BUILD_TINYOPT_BINDING  "Build python binding" OFF)
option(BUILD_TINYOPT_EXAMPLES "Build examples" OFF)
option(BUILD_TINYOPT_SOPHUS_EXAMPLES "Build Sophus examples, fecth it if not found" OFF)
option(BUILD_TINYOPT_LIEPLUSPLUS_EXAMPLES "Build Lie++ examples, fecth it if not found" OFF)
option(BUILD_TINYOPT_TESTS "Build tests" ON)
option(BUILD_TINYOPT_PACKAGES "Build packages" OFF)

option(BUILD_TINYOPT_DOC "Build documentation" OFF)
