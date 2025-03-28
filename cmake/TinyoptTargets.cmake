
if(TARGET tinyopt)
  return()
endif()

add_library(tinyopt INTERFACE IMPORTED)

set_target_properties(tinyopt PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/include"
)
