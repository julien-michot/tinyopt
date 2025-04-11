
if(NOT TARGET Tinyopt)

  add_library(tinyopt INTERFACE IMPORTED)

  # Set the include directories (assuming headers are installed in <prefix>/include)
  set_target_properties(tinyopt PROPERTIES
    INTERFACE_LINK_LIBRARIES "Eigen3::Eigen" # Link against the Eigen target
  )
endif()
