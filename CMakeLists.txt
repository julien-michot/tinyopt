cmake_minimum_required (VERSION 3.12...3.25)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(Version)

project(tinyopt VERSION ${TINYOPT_VERSION}
                DESCRIPTION "Tinyopt, a lightweight, header only optimization library"
                LANGUAGES CXX)

# Options
include(Options)

# Compiler flags
include(CompilerFlags)

# 3rd parties
include(ThirdParties)

# Header-only library
add_library(tinyopt INTERFACE)
target_include_directories(tinyopt INTERFACE include include/tinyopt/3rdparty)

# Tests
if (BUILD_TINYOPT_TESTS)
  include(Testing)
  add_subdirectory(tests)
endif ()

# Benchmarks
if (BUILD_TINYOPT_BENCHMARKS)
  add_subdirectory(benchmarks)
endif ()

#  Examples
if (BUILD_TINYOPT_EXAMPLES)
  add_subdirectory(examples)
endif ()

# Doc
if (BUILD_TINYOPT_DOCS)
  include(Documentation)
endif ()

# Installation
include(Install)

# Packaging
if (BUILD_TINYOPT_PACKAGES)
  include(Packaging)
endif()