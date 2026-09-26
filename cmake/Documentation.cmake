
# Doxygen for API documentation
find_package(Doxygen REQUIRED)

set(DOXYGEN_EXTRACT_ALL NO)
set(DOXYGEN_BUILTIN_STL_SUPPORT YES)
# Exclude implementation-only internals that trigger Sphinx/Breathe C++ parser
# crashes on complex template metaprogramming declarations in the public XML tree.
set(DOXYGEN_EXCLUDE_PATTERNS "**/3rdparty" "**/traits.h")

set(DOXYGEN_PROJECT_NUMBER ${TINYOPT_VERSION})

set(DOXYGEN_GENERATE_HTML YES)
set(DOXYGEN_GENERATE_XML YES)

# make doxygen
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in
               ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)

add_custom_target(doxygen ${DOXYGEN_EXECUTABLE}
    ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Building Doxygen documentation"
    VERBATIM)


# make docs
doxygen_add_docs(tinyopt_docs "${PROJECT_SOURCE_DIR}/include")

find_package(Sphinx REQUIRED COMPONENTS breathe)
set(SPHINX_VERSION ${PROJECT_VERSION})
sphinx_add_docs(docs
                BREATHE_PROJECTS
                    tinyopt_docs
                BUILDER
                    html
                SOURCE_DIRECTORY
                    docs/sphinx
                )
