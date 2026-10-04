# Tests

enable_testing()

function(add_test_target test_name)
  target_link_libraries(${test_name} PRIVATE tinyopt Catch2::Catch2WithMain)
  add_test(NAME ${test_name} COMMAND ${test_name})
  set_property(TEST ${test_name} PROPERTY LABELS Tinyopt Tests)
endfunction()

add_subdirectory(cpp)
add_subdirectory(c)

if(TINYOPT_BUILD_INSTALL_TESTS)
  add_test(NAME tinyopt_install_smoke_test
    COMMAND ${CMAKE_COMMAND}
      -DBUILD_TREE=${CMAKE_BINARY_DIR}
      -DPROJECT_SOURCE_DIR=${CMAKE_SOURCE_DIR}
      -DTEST_BUILD_DIR=${CMAKE_CURRENT_BINARY_DIR}/install_smoke
      -DINSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/install_prefix
      -DCMAKE_GENERATOR=${CMAKE_GENERATOR}
      -P ${CMAKE_SOURCE_DIR}/cmake/InstallSmokeTest.cmake)
endif()
