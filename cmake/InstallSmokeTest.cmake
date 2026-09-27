cmake_minimum_required(VERSION 3.25)

if(NOT DEFINED PROJECT_SOURCE_DIR OR NOT DEFINED TEST_BUILD_DIR OR NOT DEFINED INSTALL_PREFIX)
  message(FATAL_ERROR "PROJECT_SOURCE_DIR, TEST_BUILD_DIR and INSTALL_PREFIX must be defined.")
endif()

if(NOT DEFINED BUILD_TREE)
  set(BUILD_TREE "${CMAKE_BINARY_DIR}")
endif()

if(NOT DEFINED CMAKE_GENERATOR)
  set(CMAKE_GENERATOR "Ninja")
endif()

file(MAKE_DIRECTORY "${TEST_BUILD_DIR}")

execute_process(
  COMMAND "${CMAKE_COMMAND}" --install "${BUILD_TREE}" --prefix "${INSTALL_PREFIX}"
  RESULT_VARIABLE install_result
  OUTPUT_VARIABLE install_output
  ERROR_VARIABLE install_error
)

if(NOT install_result EQUAL 0)
  message(FATAL_ERROR "Failed to install Tinyopt before the smoke test.\n${install_output}\n${install_error}")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}"
    -S "${PROJECT_SOURCE_DIR}/tests/install"
    -B "${TEST_BUILD_DIR}"
    -G "${CMAKE_GENERATOR}"
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_PREFIX_PATH=${INSTALL_PREFIX}
  RESULT_VARIABLE configure_result
  OUTPUT_VARIABLE configure_output
  ERROR_VARIABLE configure_error
)

if(NOT configure_result EQUAL 0)
  message(FATAL_ERROR "Failed to configure install smoke test.\n${configure_output}\n${configure_error}")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" --build "${TEST_BUILD_DIR}"
  RESULT_VARIABLE build_result
  OUTPUT_VARIABLE build_output
  ERROR_VARIABLE build_error
)

if(NOT build_result EQUAL 0)
  message(FATAL_ERROR "Failed to build install smoke test.\n${build_output}\n${build_error}")
endif()

execute_process(
  COMMAND "${TEST_BUILD_DIR}/tinyopt_install_usage"
  RESULT_VARIABLE run_result
  OUTPUT_VARIABLE run_output
  ERROR_VARIABLE run_error
)

if(NOT run_result EQUAL 0)
  message(FATAL_ERROR "Install smoke test executable failed.\n${run_output}\n${run_error}")
endif()
