set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")

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

# Define the uninstall target
if(TARGET uninstall)
  # TODO Find fix when Eigen is Fetched...
  message(WARNING "Target '${uninstall}' already exists, skipping uninstall target.")
else()
    if(CMAKE_INSTALL_MANIFEST)
        execute_process(COMMAND ${CMAKE_COMMAND} -E echo "Removing header files listed in ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_MANIFEST}")
        add_custom_target(uninstall
                COMMAND ${CMAKE_COMMAND} -E cat "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_MANIFEST}"
                COMMAND ${CMAKE_COMMAND} -E env rm -f -- "$"
                WORKING_DIRECTORY "${CMAKE_INSTALL_PREFIX}"
                COMMENT "Uninstalling headers"
                VERBATIM
        )
    else()
        message(WARNING "CMAKE_INSTALL_MANIFEST not set, cannot create uninstall target.")
    endif()
endif()