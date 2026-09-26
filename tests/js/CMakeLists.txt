# JavaScript/WASM tests using Bun
if(NOT (EMSCRIPTEN OR CMAKE_SYSTEM_NAME STREQUAL "Emscripten"))
  message(STATUS "Skipping JavaScript tests (Emscripten not detected)")
  return()
endif()

# Find Bun test runner
set(_tinyopt_bun_hints "")
if(DEFINED ENV{BUN_INSTALL})
  list(APPEND _tinyopt_bun_hints "$ENV{BUN_INSTALL}/bin")
endif()
list(APPEND _tinyopt_bun_hints "$ENV{HOME}/.bun/bin")

find_program(BUN_EXECUTABLE bun
  HINTS ${_tinyopt_bun_hints}
)

if(BUN_EXECUTABLE)
  message(STATUS "Found Bun: ${BUN_EXECUTABLE}")

  # Add a test that runs the Bun test suite
  add_test(
    NAME js_basic_tests
    COMMAND ${BUN_EXECUTABLE} test
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )

  # Set test properties
  set_tests_properties(js_basic_tests PROPERTIES
    LABELS "Bindings;JavaScript"
    TIMEOUT 60
  )

  # Install Bun dependencies before running tests
  add_test(
    NAME js_install_deps
    COMMAND ${BUN_EXECUTABLE} install
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )

  set_tests_properties(js_install_deps PROPERTIES
    LABELS "Bindings;JavaScript"
    FIXTURES_SETUP js_deps
  )

  set_tests_properties(js_basic_tests PROPERTIES
    FIXTURES_REQUIRED js_deps
  )
else()
  message(WARNING "Bun not found - JavaScript tests will not be registered")
endif()
