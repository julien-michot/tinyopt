
# Bindings

# Find Python (required for running the code generator and building Python extension)
# For Emscripten builds, we only need the Interpreter (for code generation),
# not the Development libraries (which are for native extensions only).
if(EMSCRIPTEN OR CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
  find_package(Python 3 COMPONENTS Interpreter REQUIRED)
  message(STATUS "Building with Emscripten - skipping Python Development libraries")
else()
  find_package(Python 3 COMPONENTS Interpreter Development REQUIRED)
endif()

# nanobind is optional: build the Python module only when nanobind is available
find_package(nanobind QUIET)

if (nanobind_FOUND)
  message(STATUS "nanobind found - building Python bindings")
else()
  message(STATUS "nanobind not found - skipping Python bindings")
endif()
