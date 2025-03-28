# Installation
install(TARGETS tinyopt
        EXPORT tinyopt
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        PUBLIC_HEADER DESTINATION include/tinyopt)

# Headers
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/tinyopt/
        DESTINATION include/tinyopt
        FILES_MATCHING # install only matched files
        PATTERN "*.h"
        PATTERN "*.hpp")

# License
install(FILES LICENSE DESTINATION share/doc/tinyopt)

# Cmake
install(FILES cmake/FindTinyopt.cmake
        DESTINATION share/tinyopt/cmake)

# Doc
if(DOXYGEN_FOUND)
  install(DIRECTORY "${CMAKE_BINARY_DIR}/html/"
          DESTINATION "share/doc/tinyopt"
          FILES_MATCHING PATTERN "*")
endif()
