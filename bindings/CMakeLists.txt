
if(NOT DEFINED Python_EXECUTABLE)
  # Expect the including CMake file to have called find_package(Python ...)
  message(FATAL_ERROR "Python executable not found. Call find_package(Python 3 COMPONENTS Interpreter REQUIRED) before including bindings/CMakeLists.txt")
endif()

# Paths to generated artifacts in the build tree (preferred: keep generated
# files out of the source tree).
set(GENERATED_C_STRUCTS ${CMAKE_BINARY_DIR}/bindings/c/generated_structs.h)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/bindings/c)

# Path for generated conservative C++ conversions (written into the build tree)
set(GENERATED_C_CONVERSIONS ${CMAKE_BINARY_DIR}/bindings/c/generated_conversions.cpp)

# Path for generated Embind bindings (for JavaScript/WASM)
set(GENERATED_JS_BINDINGS ${CMAKE_BINARY_DIR}/bindings/js/generated_embind_bindings.cpp)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/bindings/js)

# Generate Python bindings (if nanobind is available)
if (nanobind_FOUND)
  set(GENERATED_PY_BINDINGS ${CMAKE_BINARY_DIR}/bindings/python/generated_struct_bindings.cpp)
  file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/bindings/python)
  set(GENERATED_C_STRUCTS2 ${CMAKE_BINARY_DIR}/bindings/c/generated_structs2.h)
  set(GENERATED_C_CONVERSIONS2 ${CMAKE_BINARY_DIR}/bindings/c/generated_conversions2.cpp)
  add_custom_command(
    OUTPUT ${GENERATED_C_STRUCTS2} ${GENERATED_PY_BINDINGS} ${GENERATED_C_CONVERSIONS2}
    COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/bindings/gen_struct_bindings.py
            ${CMAKE_SOURCE_DIR}/include/tinyopt/optimizers/options.h
            ${CMAKE_SOURCE_DIR}/include/tinyopt/output.h
            ${GENERATED_PY_BINDINGS}
            ${GENERATED_C_STRUCTS2}
            ${GENERATED_C_CONVERSIONS2}
            ${CMAKE_SOURCE_DIR}/include
    DEPENDS ${CMAKE_SOURCE_DIR}/bindings/gen_struct_bindings.py
    VERBATIM
    COMMENT "Generating C and Python bindings: ${GENERATED_C_STRUCTS2} and ${GENERATED_PY_BINDINGS} and ${GENERATED_C_CONVERSIONS2}"
  )

  add_custom_target(generate_structs2 DEPENDS ${GENERATED_C_STRUCTS} ${GENERATED_PY_BINDINGS} ${GENERATED_C_CONVERSIONS})

else()
  set(GENERATED_PY_BINDINGS "")  # No Python bindings will be generated
endif()

# Generate C bindings header and conversions
add_custom_command(
  OUTPUT ${GENERATED_C_STRUCTS} ${GENERATED_C_CONVERSIONS}
  COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/bindings/generators/gen_c_bindings.py
          ${CMAKE_SOURCE_DIR}/include/tinyopt/optimizers/options.h
          ${CMAKE_SOURCE_DIR}/include/tinyopt/output.h
          ${GENERATED_C_STRUCTS}
          ${GENERATED_C_CONVERSIONS}
          ${CMAKE_SOURCE_DIR}/include
  DEPENDS ${CMAKE_SOURCE_DIR}/bindings/generators/gen_c_bindings.py
          ${CMAKE_SOURCE_DIR}/bindings/generators/common_parser.py
          ${CMAKE_SOURCE_DIR}/bindings/generators/write_c.py
  VERBATIM
  COMMENT "Generating C bindings: ${GENERATED_C_STRUCTS} and ${GENERATED_C_CONVERSIONS}"
)

add_custom_target(generate_c_bindings DEPENDS ${GENERATED_C_STRUCTS} ${GENERATED_C_CONVERSIONS})

# Generate Embind bindings for JavaScript/WASM (separate command to avoid dependencies on Emscripten)
if (EMSCRIPTEN OR CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
  add_custom_command(
    OUTPUT ${GENERATED_JS_BINDINGS}
    COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/bindings/generators/write_embind.py
            ${CMAKE_SOURCE_DIR}/include/tinyopt/optimizers/options.h
            ${CMAKE_SOURCE_DIR}/include/tinyopt/output.h
            ${GENERATED_JS_BINDINGS}
            ${CMAKE_SOURCE_DIR}/include
    DEPENDS ${CMAKE_SOURCE_DIR}/bindings/generators/write_embind.py
            ${CMAKE_SOURCE_DIR}/bindings/generators/common_parser.py
    VERBATIM
    COMMENT "Generating Embind bindings: ${GENERATED_JS_BINDINGS}"
  )
  add_custom_target(generate_embind DEPENDS ${GENERATED_JS_BINDINGS})
endif()

## Make it easy for other targets to depend on the generated files
set(BINDINGS_GENERATED_C_STRUCTS ${GENERATED_C_STRUCTS} ${GENERATED_C_CONVERSIONS} PARENT_SCOPE)
set(BINDINGS_GENERATED_PY_BINDINGS ${GENERATED_PY_BINDINGS} PARENT_SCOPE)

# Shared tinyopt_lib target for all bindings
#
# This is an INTERFACE library for now (header-only). It provides a common
# include directory for bindings utilities, and a central place to add common
# dependencies if/when it becomes a compiled static library.
add_library(tinyopt_lib INTERFACE)
add_library(tinyopt::bindings_lib ALIAS tinyopt_lib)

target_include_directories(tinyopt_lib INTERFACE
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/bindings/include>
)

# When building via add_subdirectory(bindings) the install interface isn't
# needed, but it helps future packaging when bindings_lib becomes exported.
target_include_directories(tinyopt_lib INTERFACE
  $<INSTALL_INTERFACE:include>
)

# Depending on tinyopt makes sure downstream binding targets get the right
# include paths/compile definitions.
target_link_libraries(tinyopt_lib INTERFACE tinyopt::tinyopt)

# Include subdirectories for the bindings
message(STATUS "Adding C bindings...")
add_subdirectory(c)

if (nanobind_FOUND)
  message(STATUS "Adding Python bindings (nanobind)")
  add_subdirectory(python)
else()
  message(STATUS "nanobind not found - skipping Python bindings")
endif()

message(STATUS "Debug: CMAKE_SYSTEM_NAME is ${CMAKE_SYSTEM_NAME}")
message(STATUS "Debug: EMSCRIPTEN is ${EMSCRIPTEN}")
message(STATUS "Debug: CMAKE_TOOLCHAIN_FILE is ${CMAKE_TOOLCHAIN_FILE}")

# JavaScript/WASM bindings (Emscripten)
if (EMSCRIPTEN OR CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
  message(STATUS "Adding JavaScript/WASM bindings (Emscripten)")
  add_subdirectory(js)
else()
  message(STATUS "Emscripten not detected - skipping JavaScript bindings")
endif()