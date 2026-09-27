
if(NOT TARGET Tinyopt)

  get_filename_component(_tinyopt_prefix "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

  add_library(tinyopt INTERFACE IMPORTED)

  # Set the include directories and language requirements relative to the installed prefix.
  set_target_properties(tinyopt PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${_tinyopt_prefix}/include"
    INTERFACE_COMPILE_FEATURES "cxx_std_20"
    INTERFACE_LINK_LIBRARIES "Eigen3::Eigen"
  )
endif()
