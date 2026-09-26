
# Create the api library
# Ensure generated conversions file from the build tree (if any) is included as a source
set(_bindings_generated_files ${BINDINGS_GENERATED_C_STRUCTS})
set(_gen_conv ${CMAKE_BINARY_DIR}/bindings/c/generated_conversions.cpp)
if(NOT _gen_conv IN_LIST _bindings_generated_files)
  list(APPEND _bindings_generated_files ${_gen_conv})
endif()
# If the generated conversions file does not exist at configure time, mark it as
# GENERATED so CMake still tracks it and will re-run build rules when it appears.
if(NOT EXISTS ${_gen_conv})
  set_source_files_properties(${_gen_conv} PROPERTIES GENERATED TRUE)
endif()
add_library(tinyopt_c STATIC dense_dynamic.cpp ${_bindings_generated_files})
target_link_libraries(tinyopt_c PRIVATE tinyopt tinyopt::bindings_lib)
set_target_properties(tinyopt_c PROPERTIES
                              INTERPROCEDURAL_OPTIMIZATION OFF # OFF to speed-up compilation for now
                              CXX_VISIBILITY_PRESET default
                              VISIBILITY_INLINES_HIDDEN ON)

# Add export of all symbols for stacktrace
if (CMAKE_BUILD_TYPE MATCHES "Debug" OR CMAKE_BUILD_TYPE MATCHES "RelWithDebInfo")
  target_link_options(tinyopt_c PRIVATE -rdynamic)
endif ()

# Make sure the generated header is findable: add the build dir where the
# generator places headers.
target_include_directories(tinyopt_c PUBLIC $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/bindings/c>)

# The generated header includes the stable source header `structs.h` which
# lives in the source tree under `bindings/c`. Add the source dir to the
# include path for build time so `#include "structs.h"` resolves.
target_include_directories(tinyopt_c PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

# Ensure the C binding library is built after the generated header exists.
if(TARGET generate_c_bindings)
  add_dependencies(tinyopt_c generate_c_bindings)
endif()

# Include directories
target_include_directories(tinyopt_c INTERFACE
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
                           $<INSTALL_INTERFACE:include/c>)