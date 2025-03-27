
if(LiePlusPlus_INCLUDE_DIR)
  set(LiePlusPlus_FOUND TRUE)
else()
  find_path(LiePlusPlus_INCLUDE_DIR NAMES
            groups/SEn.hpp
            PATHS
            ${CMAKE_INSTALL_PREFIX}/include
            PATH_SUFFIXES LiePlusPlus
  )
  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(LiePlusPlus DEFAULT_MSG LiePlusPlus_INCLUDE_DIR)
  mark_as_advanced(LiePlusPlus_INCLUDE_DIR)
endif()
