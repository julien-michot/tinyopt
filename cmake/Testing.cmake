
# Tests
if (BUILD_TESTS)
  enable_testing()
  find_package(Catch2 REQUIRED)
  set (THIRDPARTY_TEST_LIBS ${THIRDPARTY_LIBS} Catch2::Catch2WithMain)

  if (${Catch2_VERSION} GREATER_EQUAL 3.0.0)
    add_definitions(-DCATCH2_VERSION=3)
  else ()
    add_definitions(-DCATCH2_VERSION=2)
  endif ()

  # FIX: add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} --verbose -T memcheck)
endif ()
