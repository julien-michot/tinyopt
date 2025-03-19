
set (THIRDPARTY_LIBS "")
set (THIRDPARTY_INCLUDE_DIRS "")

# For now, Eigen is mandatory
find_package(Eigen3 REQUIRED NO_MODULE)
if (EIGEN3_FOUND)
  message("Eigen3 found at ${EIGEN3_INCLUDE_DIR}")
  include_directories(${EIGEN3_INCLUDE_DIR})
  set (THIRDPARTY_LIBS ${THIRDPARTY_LIBS} Eigen3::Eigen)
  set (THIRDPARTY_INCLUDE_DIRS ${THIRDPARTY_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})
  add_definitions(-DHAS_EIGEN)
endif ()

if (USE_SPDLOG)
  find_package(spdlog REQUIRED)
  message("spdlog found at ${SPDLOG_INCLUDE_DIR}")
  set (THIRDPARTY_LIBS ${THIRDPARTY_LIBS} spdlog::spdlog $<$<BOOL:${MINGW}>:ws2_32>)
  add_compile_definitions(SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_TRACE)
  add_definitions(-DHAS_SPDLOG)
  set(USE_FMT OFF)
endif ()

if (USE_FMT)
  find_package(fmt REQUIRED)
  message("fmt found at ${FMT_INCLUDE_DIR}")
  set(THIRDPARTY_LIBS ${THIRDPARTY_LIBS} fmt::fmt)
  add_definitions(-DHAS_FMT)
endif ()