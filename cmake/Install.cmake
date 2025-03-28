
# Headers
install(DIRECTORY "${CMAKE_SOURCE_DIR}/include/tinyopt" # source directory
        DESTINATION "${CMAKE_INSTALL_PREFIX}/include" # target directory
        FILES_MATCHING # install only matched files
        PATTERN "*.h"
        PATTERN "*.hpp"
       )

# License
install(FILES LICENSE DESTINATION share/doc/tinyopt)

# cmake
install(FILES cmake/FindTinyopt.cmake
        DESTINATION ${CMAKE_INSTALL_PREFIX}/share/tinyopt/cmake)

# Doc
if(DOXYGEN_FOUND)
  install(DIRECTORY "${CMAKE_BINARY_DIR}/html/"
          DESTINATION "${CMAKE_INSTALL_PREFIX}/share/doc/tinyopt"
          FILES_MATCHING PATTERN "*")
endif()
