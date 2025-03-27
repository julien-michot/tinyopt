
if(Sophus_INCLUDE_DIR)
  set(Sophus_FOUND TRUE)
else()
  find_path(Sophus_INCLUDE_DIR NAMES
            sophus/se3.hpp
            PATHS
            ${CMAKE_INSTALL_PREFIX}/include
            PATH_SUFFIXES sophus
  )
  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(Sophus DEFAULT_MSG Sophus_INCLUDE_DIR)
  mark_as_advanced(Sophus_INCLUDE_DIR)
endif()
