
set(CPACK_PACKAGE_NAME "Tinyopt")

# Flags for choosing default packaging tools
set(CPACK_SOURCE_GENERATOR "TGZ" CACHE STRING "CPack Default Source Generator")
set(CPACK_GENERATOR        "TGZ" CACHE STRING "CPack Default Binary Generator")

###############################################################################
# Set up CPack
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Tinyopt, a lightweight header-only optimization library")
set(CPACK_PACKAGE_VENDOR "Julien Michot")
set(CPACK_PACKAGE_CONTACT "julien.michot.fr@gmail.com")
set(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_PACKAGE_VERSION_MAJOR ${TINYOPT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${TINYOPT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${TINYOPT_VERSION_PATCH})
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_SOURCE_IGNORE_FILES "/build*;")
set(CPACK_SOURCE_PACKAGE_FILE_NAME "TINYOPT-${TINYOPT_VERSION_STRING}")

# Deb-package specific cpack
set(CPACK_DEBIAN_PACKAGE_NAME "libtinyopt-dev")
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libeigen3-dev (>= 3.0)")
set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA "Apache 2.0")

# RPM
set(CPACK_RPM_PACKAGE_NAME "libtinyopt-dev")
set(CPACK_RPM_PACKAGE_REQUIRES "libeigen3-dev (>= 3.0)")
set(CPACK_RPM_PACKAGE_LICENSE "Apache 2.0")

include(CPack)
