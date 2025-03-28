# FindTinyopt.cmake - Find Tinyopt library.
#
# Defines the following variables:
#  Tinyopt_FOUND - True if Tinyopt was found.
#  Tinyopt_INCLUDE_DIRS - Include directories for Tinyopt.

find_path(Tinyopt_INCLUDE_DIRS
  NAMES tinyopt.h # Or whatever your header file is called.
  PATHS
  /usr/local/include
  /usr/include
  ${CMAKE_SOURCE_DIR}/include
  PATH_SUFFIXES tinyopt
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Tinyopt
  REQUIRED_VARS Tinyopt_INCLUDE_DIRS
)

mark_as_advanced(Tinyopt_INCLUDE_DIRS)

if(Tinyopt_FOUND)
  message(STATUS "Found Tinyopt at ${Tinyopt_INCLUDE_DIRS}")
endif()
