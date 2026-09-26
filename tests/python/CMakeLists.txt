
# Basic test

add_test(NAME Python.Basic
         COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/python/test_basic.py)
set_tests_properties(Python.Basic PROPERTIES LABELS "Tinyopt;Python;Bindings;Tests" ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/bindings/python")

# Manifold test
add_test(NAME Python.Manifold
         COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/python/test_manifold.py)
set_tests_properties(Python.Manifold PROPERTIES LABELS "Tinyopt;Python;Bindings;Tests" ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/bindings/python")
