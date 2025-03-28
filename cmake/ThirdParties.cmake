
set (THIRDPARTY_LIBS "")
set (THIRDPARTY_INCLUDE_DIRS "")

include(FetchContent)

# For now, Eigen is mandatory
find_package(Eigen3 REQUIRED NO_MODULE)
if (EIGEN3_FOUND)
  message("Eigen3 found at ${EIGEN3_INCLUDE_DIR}")
  set (THIRDPARTY_LIBS ${THIRDPARTY_LIBS} Eigen3::Eigen)
  set (THIRDPARTY_INCLUDE_DIRS ${THIRDPARTY_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})
  add_definitions(-DHAS_EIGEN)
endif ()

if (USE_FMT)
  find_package(fmt REQUIRED)
  message("fmt found at ${FMT_INCLUDE_DIR}")
  set(THIRDPARTY_LIBS ${THIRDPARTY_LIBS} fmt::fmt)
  add_definitions(-DHAS_FMT)
endif ()

if (BUILD_TINYOPT_SOPHUS_EXAMPLES)
  find_package(Sophus)
  if (NOT Sophus_FOUND)
    message("Sophus not found, fetching...")
    set(SOPHUS_FIX_CMAKE_VER sed -i -E "s/3.24/3.2/g" CMakeLists.txt)
    FetchContent_Declare(Sophus
                         GIT_REPOSITORY https://github.com/strasdat/Sophus.git
                         GIT_TAG main
                         PATCH_COMMAND ${SOPHUS_FIX_CMAKE_VER}
                         UPDATE_DISCONNECTED 1
    )
    FetchContent_MakeAvailable(Sophus)
  endif ()
  add_definitions(-DHAS_SOPHUS)
  #include_directories(${Sophus_SOURCE_DIR}/sophus)
  #add_definitions(-DSOPHUS_USE_BASIC_LOGGING=1)
  set (THIRDPARTY_LIBS ${THIRDPARTY_LIBS} Sophus::Sophus)
  set (THIRDPARTY_INCLUDE_DIRS ${THIRDPARTY_INCLUDE_DIRS} ${SOPHUS_INCLUDE_DIR})
endif ()


if (BUILD_TINYOPT_LIEPLUSPLUS_EXAMPLES)
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
    FetchContent_MakeAvailable(LiePlusPlus)
  endif ()
  add_definitions(-DHAS_LIEPLUSPLUS)
  #include_directories(${LiePlusPlus_SOURCE_DIR}/include)
  set (THIRDPARTY_LIBS ${THIRDPARTY_LIBS} LiePlusPlus)
  set (THIRDPARTY_INCLUDE_DIRS ${THIRDPARTY_INCLUDE_DIRS} ${SOPHUS_INCLUDE_DIR})
endif ()