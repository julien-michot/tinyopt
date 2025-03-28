
# Tests
enable_testing()

find_package(Catch2)
if (NOT Catch2_FOUND)

  include(FetchContent)

  message("Catch2 is missing, fetching...")

  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY  https://github.com/catchorg/Catch2.git
    GIT_TAG         devel
    GIT_SHALLOW     TRUE
    GIT_PROGRESS    TRUE
  )
  FetchContent_MakeAvailable(Catch2)
endif ()
set(THIRDPARTY_TEST_LIBS ${THIRDPARTY_LIBS} Catch2::Catch2WithMain)

if (${Catch2_VERSION} GREATER_EQUAL 3.0.0)
  add_definitions(-DCATCH2_VERSION=3)
else ()
  add_definitions(-DCATCH2_VERSION=2)
endif ()

# FIX: add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} --verbose -T memcheck)