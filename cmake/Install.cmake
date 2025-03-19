
# Headers
install(DIRECTORY "${CMAKE_SOURCE_DIR}/include/tinyopt" # source directory
        DESTINATION "include" # target directory
        FILES_MATCHING # install only matched files
        PATTERN "*.h"
        PATTERN "*.hpp"
       )

# TODO Apps (examples), cmake, doc, ...