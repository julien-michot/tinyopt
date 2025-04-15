
set(THIRDPARTY_LIBS "")
set(THIRDPARTY_INCLUDE_DIRS "")

include(FetchContent)

# For now, Eigen is mandatory
find_package(Eigen3 NO_MODULE)
if (EIGEN3_FOUND)
  message("Eigen3 found at ${EIGEN3_INCLUDE_DIR}")
else()
  message("Eigen3 is missing, fetching...")
  FetchContent_Declare(
    Eigen
    GIT_REPOSITORY  https://gitlab.com/libeigen/eigen.git
    GIT_TAG         3.4.0
    GIT_SHALLOW     TRUE
    GIT_PROGRESS    TRUE
  )
  block (SCOPE_FOR VARIABLES) # requires cmake 3.25+
    set(BUILD_TESTING OFF)
    set(EIGEN_BUILD_TESTING OFF)
    set(EIGEN_TEST_CXX11 OFF)
    set(EIGEN_HAS_CXX11_MATH ON)
    set(EIGEN_BUILD_DOC OFF)
    set(EIGEN_BUILD_PKGCONFIG OFF)
    FetchContent_MakeAvailable(Eigen)
  endblock ()
endif ()
add_definitions(-DHAS_EIGEN)
set(THIRDPARTY_LIBS ${THIRDPARTY_LIBS} Eigen3::Eigen)
set(THIRDPARTY_INCLUDE_DIRS ${THIRDPARTY_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})

if (TINYOPT_USE_FMT)
  find_package(fmt REQUIRED)
  message("fmt found at ${FMT_INCLUDE_DIR}")
  set(THIRDPARTY_LIBS ${THIRDPARTY_LIBS} fmt::fmt)
  add_definitions(-DHAS_FMT)
endif ()


if (TINYOPT_BUILD_CERES)
  find_package(Ceres)
  if (NOT Ceres_FOUND)
    message("Ceres not found, fetching...")
    FetchContent_Declare(Ceres
                         GIT_REPOSITORY https://github.com/ceres-solver/ceres-solver
                         GIT_TAG 2.2.0
                         GIT_SHALLOW     TRUE
                         GIT_PROGRESS    TRUE)
    set(BUILD_TESTING OFF)
    set(BUILD_EXAMPLES OFF)
    set(BUILD_BENCHMARKS OFF)
    FetchContent_MakeAvailable(Ceres)
    set(CERES_LIBRARIES Ceres::ceres)
    target_compile_options(ceres PUBLIC "-Wno-reorder" "-Wno-maybe-uninitialized")

  endif ()
  set(THIRDPARTY_INCLUDE_DIRS ${THIRDPARTY_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS})
  set(THIRDPARTY_LIBS ${THIRDPARTY_LIBS} ${CERES_LIBRARIES})
  add_definitions(-DHAS_CERES)
endif()


if (TINYOPT_BUILD_SOPHUS_TEST)
  find_package(Sophus)
  if (NOT Sophus_FOUND)
    message("Sophus not found, fetching...")
    set(SOPHUS_FIX_CMAKE_VER sed -i -E "s/3.24/3.2/g" CMakeLists.txt)
    FetchContent_Declare(Sophus
                         GIT_REPOSITORY https://github.com/strasdat/Sophus.git
                         GIT_TAG main
                         GIT_SHALLOW     TRUE
                         GIT_PROGRESS    TRUE
                         PATCH_COMMAND ${SOPHUS_FIX_CMAKE_VER}
                         UPDATE_DISCONNECTED 1
    )
    set(BUILD_SOPHUS_TESTS OFF)
    FetchContent_MakeAvailable(Sophus)
  endif ()
  add_definitions(-DHAS_SOPHUS)
  #include_directories(${Sophus_SOURCE_DIR}/sophus)
  #add_definitions(-DSOPHUS_USE_BASIC_LOGGING=1)
  set(THIRDPARTY_LIBS ${THIRDPARTY_LIBS} Sophus::Sophus)
  set(THIRDPARTY_INCLUDE_DIRS ${THIRDPARTY_INCLUDE_DIRS} ${SOPHUS_INCLUDE_DIR})
endif ()


if (TINYOPT_BUILD_LIEPLUSPLUS_TEST)
  find_package(LiePlusPlus)
  if (NOT LiePlusPlus_FOUND)
    message("Lie++ not found, fetching...")
    FetchContent_Declare(
        LiePlusPlus
        GIT_REPOSITORY  https://github.com/julien-michot/Lie-plusplus
        GIT_TAG         main
        GIT_SHALLOW     TRUE
        GIT_PROGRESS    TRUE
    )
    set(LIEPLUSPLUS_TESTS OFF)
    FetchContent_MakeAvailable(LiePlusPlus)
  endif ()
  add_definitions(-DHAS_LIEPLUSPLUS)
  #include_directories(${LiePlusPlus_SOURCE_DIR}/include)
  set(THIRDPARTY_LIBS ${THIRDPARTY_LIBS} LiePlusPlus)
  set(THIRDPARTY_INCLUDE_DIRS ${THIRDPARTY_INCLUDE_DIRS} ${SOPHUS_INCLUDE_DIR})
endif ()


if (TINYOPT_BUILD_TESTS OR TINYOPT_BUILD_BENCHMARKS)
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
endif()