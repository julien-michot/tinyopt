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

# Doc
if(DOXYGEN_FOUND)
  install(DIRECTORY "${CMAKE_BINARY_DIR}/html/"
          DESTINATION "share/doc/tinyopt"
          FILES_MATCHING PATTERN "*")
endif()

# Cmake
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/TinyoptConfig.cmake.in
    ${CMAKE_BINARY_DIR}/TinyoptConfig.cmake
    @ONLY
)

# Install the Config file.
install(FILES ${CMAKE_BINARY_DIR}/TinyoptConfig.cmake
              ${CMAKE_CURRENT_SOURCE_DIR}/cmake/TinyoptTargets.cmake
        DESTINATION lib/cmake/tinyopt
)
install(FILES cmake/FindTinyopt.cmake
        DESTINATION lib/cmake/tinyopt)