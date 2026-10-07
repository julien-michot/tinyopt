set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")

# Installation
install(TARGETS tinyopt
        EXPORT tinyopt
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        PUBLIC_HEADER DESTINATION include/tinyopt)

if(TINYOPT_BUILD_C_LIBRARY)
  install(TARGETS tinyopt_c
          RUNTIME DESTINATION bin
          LIBRARY DESTINATION lib
          ARCHIVE DESTINATION lib)
endif()

# Headers
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/tinyopt/
        DESTINATION include/tinyopt
        FILES_MATCHING # install only matched files
        PATTERN "*.h"
        PATTERN "*.hpp")

install(DIRECTORY "${TINYOPT_C_FIXED_INCLUDE_DIR}/tinyopt/c/"
        DESTINATION include/tinyopt/c
        FILES_MATCHING PATTERN "*.h")

# License
install(FILES LICENSE DESTINATION share/doc/tinyopt)
install(FILES ${CMAKE_SOURCE_DIR}/include/tinyopt/3rdparty/ceres/LICENSE
        DESTINATION share/doc/tinyopt/3rdparty/ceres)

# Documentation
if(TINYOPT_BUILD_DOCS)
  install(DIRECTORY "${CMAKE_BINARY_DIR}/html/"
          DESTINATION "share/doc/tinyopt"
          FILES_MATCHING PATTERN "*")
  install(DIRECTORY "${CMAKE_BINARY_DIR}/docs/"
          DESTINATION "share/doc/tinyopt/guide"
          FILES_MATCHING PATTERN "*"
          PATTERN "examples.html" EXCLUDE
          PATTERN "wasm-example.html" EXCLUDE
          PATTERN "wasm-tutorial.html" EXCLUDE
          PATTERN "examples.rst.txt" EXCLUDE
          PATTERN "wasm-example.rst.txt" EXCLUDE
          PATTERN "wasm-tutorial.rst.txt" EXCLUDE
          PATTERN "_downloads" EXCLUDE
          PATTERN "tinyopt_wasm.png" EXCLUDE)
endif()

# CMake package metadata
include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/TinyoptConfig.cmake.in
    ${CMAKE_BINARY_DIR}/TinyoptConfig.cmake
    INSTALL_DESTINATION lib/cmake/Tinyopt
)
write_basic_package_version_file(
    ${CMAKE_BINARY_DIR}/TinyoptConfigVersion.cmake
    VERSION ${TINYOPT_VERSION_STRING}
    COMPATIBILITY SameMajorVersion)

# Install the config files in the standard CMake package location.
install(FILES ${CMAKE_BINARY_DIR}/TinyoptConfig.cmake
              ${CMAKE_BINARY_DIR}/TinyoptConfigVersion.cmake
              ${CMAKE_CURRENT_SOURCE_DIR}/cmake/TinyoptTargets.cmake
        DESTINATION lib/cmake/Tinyopt
)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindTinyopt.cmake
        DESTINATION lib/cmake/Tinyopt)

# Keep a lowercase alias for compatibility with older package layout expectations.
install(FILES ${CMAKE_BINARY_DIR}/TinyoptConfig.cmake
              ${CMAKE_BINARY_DIR}/TinyoptConfigVersion.cmake
              ${CMAKE_CURRENT_SOURCE_DIR}/cmake/TinyoptTargets.cmake
        DESTINATION lib/cmake/tinyopt)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindTinyopt.cmake
        DESTINATION lib/cmake/tinyopt)

# Define the uninstall target
if(TARGET uninstall)
  # TODO Find fix when Eigen is Fetched...
  message(WARNING "Target 'uninstall' already exists, skipping uninstall target.")
else()
    if(CMAKE_INSTALL_MANIFEST)
        add_custom_target(uninstall COMMENT "Uninstall installed files")
        add_custom_command(
        TARGET uninstall
        POST_BUILD
        COMMENT "Uninstall files with install_manifest.txt"
        COMMAND xargs rm -vf < install_manifest.txt || echo Nothing in
                install_manifest.txt to be uninstalled!
        COMMAND rm -fr ${CMAKE_INSTALL_PREFIX}/include/tinyopt
                       ${CMAKE_INSTALL_PREFIX}/share/doc/tinyopt
                       ${CMAKE_INSTALL_PREFIX}/lib/cmake/tinyopt ||
                echo ${CMAKE_INSTALL_PREFIX} does not contain tinyopt
        )
    else()
        message(WARNING "CMAKE_INSTALL_MANIFEST not set, cannot create uninstall target.")
    endif()
endif()