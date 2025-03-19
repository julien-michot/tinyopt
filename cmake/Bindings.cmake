
# Bindings
if (BUILD_BINDING)
  find_package(Python 3.10 COMPONENTS Interpreter Development)
  find_package(pybind11 CONFIG)

  # Create binding
  add_definitions(-DPYBIND11_DETAILED_ERROR_MESSAGES)
  Python_add_library(tinyopt "")
  target_link_libraries(tinyopt PUBLIC pybind11::headers ${THIRDPARTY_LIBS})
  set_target_properties(tinyopt PROPERTIES
                                INTERPROCEDURAL_OPTIMIZATION OFF # OFF to speed-up compilation for now
                                CXX_VISIBILITY_PRESET default
                                VISIBILITY_INLINES_HIDDEN ON)

  # Add export of all symbols for stacktrace
  if (CMAKE_BUILD_TYPE MATCHES "Debug" OR CMAKE_BUILD_TYPE MATCHES "RelWithDebInfo")
    target_link_options(tinyopt PRIVATE -rdynamic)
  endif ()

endif()
