#find_package(Python 3 COMPONENTS Interpreter Development REQUIRED)
#find_package(nanobind REQUIRED)

# Include common bindings generation steps (generates C + Python binding glue)
#include(${CMAKE_SOURCE_DIR}/bindings/CMakeLists.txt)

nanobind_add_module(tinyopt_py ${CMAKE_CURRENT_SOURCE_DIR}/python_bindings.cpp ${BINDINGS_GENERATED_PY_BINDINGS})
target_link_libraries(tinyopt_py PRIVATE tinyopt::tinyopt tinyopt::bindings_lib)

target_include_directories(tinyopt_py PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
)

# Ensure the main project headers are visible (include/tinyopt/...)
target_include_directories(tinyopt_py PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Ensure Python headers are visible when compiling the binding. The FindPython
# call above exposes Python_INCLUDE_DIRS and a Python::Python target; add the
# include directory to the module target so nanobind headers can find Python.h.
if (DEFINED Python_INCLUDE_DIRS)
  target_include_directories(tinyopt_py PRIVATE ${Python_INCLUDE_DIRS})
endif()

if (DEFINED nanobind_SOURCE_DIR)
  # If we fetched nanobind via FetchContent, ensure its include dir is visible.
  target_include_directories(tinyopt_py PRIVATE ${nanobind_SOURCE_DIR}/include)
endif()

# Ensure generated files directory is an include path for the target
target_include_directories(tinyopt_py PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

# Ensure generated files are present before building the python module
add_dependencies(tinyopt_py generate_structs2)

set_target_properties(tinyopt_py PROPERTIES
  PREFIX ""
  CXX_STANDARD 20
  POSITION_INDEPENDENT_CODE ON
  OUTPUT_NAME "tinyopt"
)

if (CMAKE_BUILD_TYPE MATCHES "Debug" OR CMAKE_BUILD_TYPE MATCHES "RelWithDebInfo")
  target_link_options(tinyopt_py PRIVATE -rdynamic)
endif()
