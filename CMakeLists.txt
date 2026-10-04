cmake_minimum_required (VERSION 3.25)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(Version)

project(tinyopt VERSION ${TINYOPT_VERSION}
                DESCRIPTION "Tinyopt, a lightweight, header only optimization library"
                LANGUAGES C CXX)

# Options
include(Options)
include(Configuration)
include(GenerateFixedCAPI)

# Compiler flags
include(CompilerFlags)

# Header-only library
add_library(tinyopt INTERFACE)

# 3rd parties
include(ThirdParties)

# Configure
configure_tinyopt_features(tinyopt)

target_include_directories(tinyopt INTERFACE
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                           $<INSTALL_INTERFACE:include>)

# C ABI library for C applications and language bindings.
add_library(tinyopt_c SHARED
            src/opt_dyn.cpp
            ${TINYOPT_C_GENERATED_WRAPPERS}
            ${TINYOPT_C_GENERATED_BACKENDS})
target_link_libraries(tinyopt_c PRIVATE tinyopt)
target_include_directories(tinyopt_c PUBLIC
                           $<BUILD_INTERFACE:${TINYOPT_C_FIXED_INCLUDE_DIR}>
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                           $<INSTALL_INTERFACE:include>
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_definitions(tinyopt_c PRIVATE TINYOPT_C_BUILD)
set_target_properties(tinyopt_c PROPERTIES
                      CXX_VISIBILITY_PRESET hidden
                      VISIBILITY_INLINES_HIDDEN ON)

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

# Python package installation
if (TINYOPT_BUILD_PIP_PACKAGE)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  add_custom_target(pip-install
    COMMAND "${Python3_EXECUTABLE}" -m pip install --no-deps "${CMAKE_CURRENT_SOURCE_DIR}"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Installing Tinyopt with pip"
    VERBATIM)
endif()
