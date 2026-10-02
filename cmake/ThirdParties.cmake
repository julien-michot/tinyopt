
include(FetchContent)

set(BUILD_TESTING_OLD ${BUILD_TESTING}) # Save setting
set(BUILD_TESTING OFF CACHE BOOL "" FORCE) # Disable third party tests

# For now, Eigen is mandatory
find_package(Eigen3 QUIET)
set(EIGEN_BUILD_TESTING OFF)
set(EIGEN_BUILD_DOC OFF)
set(EIGEN_BUILD_PKGCONFIG OFF)
if (Eigen3_FOUND)
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
    set(EIGEN_TEST_CXX11 OFF)
    set(EIGEN_HAS_CXX11_MATH ON)
    FetchContent_MakeAvailable(Eigen)
  endblock ()
  set(EIGEN3_INCLUDE_DIR "${eigen3_SOURCE_DIR}" CACHE PATH "Eigen3 include directory" FORCE)
  set(Eigen3_FOUND TRUE CACHE BOOL "Eigen3 found" FORCE)
endif ()

if (TINYOPT_ENABLE_SUITESPARSE)
  enable_language(C)
  find_package(CHOLMOD CONFIG REQUIRED)
endif ()

# Eigen is mandatory
if (NOT TARGET Eigen3::Eigen)
  message(FATAL_ERROR "Eigen3 not found")
endif ()


# FMT
if (TINYOPT_USE_FMT)
  find_package(fmt REQUIRED)
  message("fmt found at ${FMT_INCLUDE_DIR}")
endif ()


if (TINYOPT_BUILD_CERES)
  enable_language(C)
  find_package(Ceres QUIET)
  set(MINIGLOG ON)
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
    set(MINIGLOG ON)
    FetchContent_MakeAvailable(Ceres)
    target_compile_options(ceres PUBLIC "-Wno-reorder" "-Wno-maybe-uninitialized")
  endif ()
  if(NOT TARGET Ceres::ceres)
    message(FATAL_ERROR "Ceres target not found")
  endif()
endif()

if (TINYOPT_BUILD_G2O_BENCHMARKS)
  find_package(g2o CONFIG QUIET)
  if (NOT TARGET g2o::core AND NOT TARGET g2o_core)
    message("g2o package not found, fetching g2o from source...")
    set(G2O_BUILD_APPS OFF CACHE BOOL "" FORCE)
    set(G2O_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(G2O_BUILD_UNITTESTS OFF CACHE BOOL "" FORCE)
    set(G2O_BUILD_SLAM2D OFF CACHE BOOL "" FORCE)
    set(G2O_BUILD_SLAM3D OFF CACHE BOOL "" FORCE)
    set(G2O_BUILD_SBA ON CACHE BOOL "" FORCE)
    set(G2O_USE_OPENGL OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
      g2o
      GIT_REPOSITORY https://github.com/RainerKuemmerle/g2o.git
      GIT_TAG fd65627348213d56866b24c0b7d054bcf105f0a5
      GIT_PROGRESS TRUE
    )
    FetchContent_MakeAvailable(g2o)
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
      foreach(g2o_core_target IN ITEMS g2o::core g2o_core core)
        if (TARGET ${g2o_core_target})
          get_target_property(g2o_real_core_target ${g2o_core_target} ALIASED_TARGET)
          if (g2o_real_core_target)
            set(g2o_core_target ${g2o_real_core_target})
          endif()
          target_compile_options(${g2o_core_target} PRIVATE -Wno-error=unused-but-set-variable)
          break()
        endif()
      endforeach()
    endif()
  endif()
  if (NOT TARGET g2o::core AND NOT TARGET g2o_core)
    message(FATAL_ERROR "g2o core target not found")
  endif()
endif()

if (TINYOPT_BUILD_GTSAM_BENCHMARKS)
  find_package(GTSAM REQUIRED)
endif()


if (TINYOPT_BUILD_SOPHUS_TEST)
  find_package(Sophus QUIET)
  if (NOT Sophus_FOUND)
    message("Sophus not found, fetching...")
    set(SOPHUS_FIX_CMAKE_VER sed -i -E "s/3.24/3.5/g" CMakeLists.txt)
    FetchContent_Declare(Sophus
                         GIT_REPOSITORY https://github.com/strasdat/Sophus.git
                         GIT_TAG main
                         GIT_SHALLOW     TRUE
                         GIT_PROGRESS    TRUE
                         PATCH_COMMAND ${SOPHUS_FIX_CMAKE_VER}
                         UPDATE_DISCONNECTED 1
    )
    block (SCOPE_FOR VARIABLES) # requires cmake 3.25+
      set(BUILD_SOPHUS_TESTS OFF)
      FetchContent_MakeAvailable(Sophus)
    endblock ()
  endif ()
  message("Sophus ${Sophus_FOUND} : found at ${Sophus_INCLUDE_DIR}")

  if(TARGET Sophus::Sophus)
  elseif(Sophus_FOUND)
  else()
    message(FATAL_ERROR "Sophus target not found")
  endif ()
endif ()


if (TINYOPT_BUILD_LIEPLUSPLUS_TEST)
  find_package(LiePlusPlus QUIET)
  set(LIEPLUSPLUS_TESTS OFF)
  if (NOT LiePlusPlus_FOUND)
    message("Lie++ not found, fetching...")
    FetchContent_Declare(
        LiePlusPlus
        GIT_REPOSITORY  https://github.com/julien-michot/Lie-plusplus
        GIT_TAG         main
        GIT_SHALLOW     TRUE
        GIT_PROGRESS    TRUE
    )
    block (SCOPE_FOR VARIABLES) # requires cmake 3.25+
      set(LIEPLUSPLUS_TESTS OFF)
      FetchContent_MakeAvailable(LiePlusPlus)
    endblock ()
    FetchContent_MakeAvailable(LiePlusPlus)
  endif ()
  message("LiePlusPlus ${LiePlusPlus_FOUND} : found at ${LiePlusPlus_INCLUDE_DIR} target ${LiePlusPlus_TARGET}")

  # Conditionally link to LiePlusPlus
  if(NOT LiePlusPlus_INCLUDE_DIR)
    message(FATAL_ERROR "LiePlusPlus not found")
  endif()
endif ()


if (TINYOPT_BUILD_TESTS OR TINYOPT_BUILD_BENCHMARKS)
  find_package(Catch2 3 REQUIRED)
  # Conditionally link to Catch2
  if(TARGET Catch2::Catch2WithMain)
    if (${Catch2_VERSION} GREATER_EQUAL 3.0.0)
        set(CATCH2_MAJOR_VERSION 3)
    else ()
        set(CATCH2_MAJOR_VERSION 2)
    endif ()
  else()
    message(FATAL_ERROR "Catch2 target not found")
  endif()

endif()
set(BUILD_TESTING ${BUILD_TESTING_OLD} CACHE BOOL "" FORCE) # Restore testing config
