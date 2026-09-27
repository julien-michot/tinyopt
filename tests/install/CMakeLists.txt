cmake_minimum_required(VERSION 3.25)
project(tinyopt_install_smoke LANGUAGES CXX)

list(GET CMAKE_PREFIX_PATH 0 _tinyopt_prefix)
set(TinyoptConfigFile "${_tinyopt_prefix}/lib/cmake/Tinyopt/TinyoptConfig.cmake")

if(NOT EXISTS "${TinyoptConfigFile}")
  message(FATAL_ERROR "Tinyopt package config not found at ${TinyoptConfigFile}")
endif()

include("${TinyoptConfigFile}")

add_executable(tinyopt_install_usage main.cpp)
target_link_libraries(tinyopt_install_usage PRIVATE tinyopt)
