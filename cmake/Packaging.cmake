
set(CPACK_PACKAGE_NAME ${CMAKE_PROJECT_NAME})

# Flags for choosing default packaging tools
set(CPACK_SOURCE_GENERATOR "TGZ" CACHE STRING "CPack Default Source Generator")
set(CPACK_GENERATOR        "TGZ" CACHE STRING "CPack Default Binary Generator")

###############################################################################
# Set up CPack
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Tinyopt, a lightweight header-only optimization library")
set(CPACK_DEFAULT_PACKAGE_DESCRIPTION_SUMMARY ${CPACK_PACKAGE_DESCRIPTION_SUMMARY})
set(CPACK_PACKAGE_VENDOR "Julien Michot")
set(CPACK_PACKAGE_CONTACT "julien.michot.fr@gmail.com")
set(CPACK_PACKAGE_HOMEPAGE_URL "https://github.com/julien-michot/tinyopt")
set(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/README.txt")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_PACKAGE_VERSION_MAJOR ${TINYOPT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${TINYOPT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${TINYOPT_VERSION_PATCH})
set(CPACK_PACKAGE_VERSION ${TINYOPT_VERSION_STRING})
set(CPACK_SOURCE_IGNORE_FILES "/build*;.git*;.cache;.vscode;b/*")
set(CPACK_SOURCE_PACKAGE_FILE_NAME "${CMAKE_PROJECT_NAME}-${TINYOPT_VERSION_STRING}")

# Deb-package specific cpack
set(CPACK_DEBIAN_PACKAGE_NAME "libtinyopt-dev")
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libeigen3-dev (>= 3.0)")
set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")

# RPM
set(CPACK_RPM_PACKAGE_NAME "libtinyopt-dev")
set(CPACK_RPM_PACKAGE_REQUIRES "libeigen3-dev (>= 3.0)")
set(CPACK_RPM_PACKAGE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")

add_custom_target(src
    COMMAND ${CMAKE_CPACK_COMMAND} --config CPackSourceConfig.cmake
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Building Source package"
    VERBATIM)

add_custom_target(deb
    COMMAND ${CMAKE_CPACK_COMMAND} -G DEB
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Building Debian package"
    VERBATIM)

include(CPack)
