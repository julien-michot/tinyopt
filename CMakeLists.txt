cmake_minimum_required (VERSION 3.25)

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
target_include_directories(tinyopt INTERFACE include include/tinyopt/3rdparty ${THIRDPARTY_INCLUDE_DIRS})
set_property(TARGET tinyopt PROPERTY INTERFACE_LINK_LIBRARIES ${THIRDPARTY_LIBS})

# Tests
if (TINYOPT_BUILD_TESTS)
  include(Testing)
  add_subdirectory(tests)
endif ()

# Benchmarks
if (TINYOPT_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif ()

#  Examples
if (TINYOPT_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif ()

# Doc
if (TINYOPT_BUILD_DOCS)
  include(Documentation)
endif ()

# Installation
include(Install)

# Packaging
if (TINYOPT_BUILD_PACKAGES)
  include(Packaging)
endif()