# JavaScript/WASM bindings using Emscripten

message(STATUS "Building JavaScript/WASM bindings with Emscripten")

# Create the WASM module using Embind
add_executable(tinyopt_js js_bindings.cpp)

# Shared bindings utilities
target_link_libraries(tinyopt_js PRIVATE tinyopt::bindings_lib)

target_include_directories(tinyopt_js PRIVATE
  ${CMAKE_SOURCE_DIR}/bindings/include
)

if (EMSCRIPTEN OR CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
  # Some environments propagate default link libs (e.g., from test setup).
  # The wasm module should only link what it directly needs.
  set_target_properties(tinyopt_js PROPERTIES LINK_LIBRARIES "")
endif()

# IMPORTANT (Emscripten): the JS/WASM module must not link against host-side
# libraries like Catch2. Tests are executed separately via Bun/CTest.
# Ensure no Catch2 libraries are pulled in transitively.
if (TARGET Catch2::Catch2WithMain)
  set_property(TARGET tinyopt_js PROPERTY INTERFACE_LINK_LIBRARIES "")
  target_link_libraries(tinyopt_js PRIVATE $<LINK_ONLY:>)
endif()

# Don't link against tinyopt::tinyopt to avoid pulling in conda libraries
# Instead, just use the include directories
target_include_directories(tinyopt_js PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIR}
)

# Ensure generated files directory is an include path
target_include_directories(tinyopt_js PRIVATE ${CMAKE_BINARY_DIR}/bindings/js)

# Ensure generated Embind bindings are present before building
if(TARGET generate_embind)
  add_dependencies(tinyopt_js generate_embind)
endif()

set_target_properties(tinyopt_js PROPERTIES
  CXX_STANDARD 20
  OUTPUT_NAME "tinyopt"
  LINK_FLAGS ""  # Clear any inherited link flags from conda
  LINK_DIRECTORIES ""  # Clear link directories
)

# Clear macOS-specific linker flags that conda adds but wasm-ld doesn't understand
set(CMAKE_EXE_LINKER_FLAGS "")
set(CMAKE_SHARED_LINKER_FLAGS "")
set(CMAKE_MODULE_LINKER_FLAGS "")

# Clear conda's C++ flags that interfere with emscripten
string(REPLACE "-stdlib=libc++" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

# Explicitly tell CMake not to add conda library paths
set(CMAKE_SKIP_RPATH TRUE)
set(CMAKE_SKIP_BUILD_RPATH TRUE)
set(CMAKE_SKIP_INSTALL_RPATH TRUE)

# Remove conda library directories from the link path
list(REMOVE_ITEM CMAKE_SYSTEM_LIBRARY_PATH "$ENV{CONDA_PREFIX}/lib")
list(REMOVE_ITEM CMAKE_SYSTEM_PREFIX_PATH "$ENV{CONDA_PREFIX}")

# Clear any conda library paths that might interfere with emscripten
set_target_properties(tinyopt_js PROPERTIES
  LINK_DIRECTORIES ""
  INTERFACE_LINK_DIRECTORIES ""
)

# Emscripten-specific compile options
target_compile_options(tinyopt_js PRIVATE
  -fexceptions  # Enable exception support
  -fno-stack-protector  # Disable stack protection for WASM
)

# Clear any conda library paths that might interfere with emscripten
set_target_properties(tinyopt_js PROPERTIES
  LINK_DIRECTORIES ""
  INTERFACE_LINK_DIRECTORIES ""
)

# Emscripten link flags for binding and WASM output
# Use SHELL: prefix to pass flags exactly as-is to emcc
target_link_options(tinyopt_js PRIVATE
  "SHELL:--bind"                           # Enable Embind
  "SHELL:-fexceptions"                     # Enable exception handling
  "SHELL:-sALLOW_MEMORY_GROWTH=1"          # Allow memory to grow
  "SHELL:-sMODULARIZE=1"                   # Create a factory function
  "SHELL:-sEXPORT_ES6=1"                   # Export as ES6 module
  "SHELL:-sEXPORT_NAME=createTinyoptModule" # Factory function name
  "SHELL:-sENVIRONMENT=node,web"           # Support both Node.js and browser
  "SHELL:-sEXPORTED_RUNTIME_METHODS=[\"ccall\",\"cwrap\"]"
  "SHELL:-sWASM=1"                         # Output WASM (default, but explicit)
  "SHELL:-O3"                              # Optimize for performance
)

# Optionally enable debug symbols in Debug builds
if(CMAKE_BUILD_TYPE MATCHES "Debug")
  target_link_options(tinyopt_js PRIVATE
    -g                             # Include debug info
    -sASSERTIONS=1                 # Enable runtime assertions
    -sSAFE_HEAP=1                  # Memory safety checks
  )
endif()

# Install the generated WASM and JS files
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/tinyopt.js
  ${CMAKE_CURRENT_BINARY_DIR}/tinyopt.wasm
  DESTINATION lib/js
  OPTIONAL
)

# Make the output available in the build tree for testing
add_custom_command(TARGET tinyopt_js POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_BINARY_DIR}/tinyopt.js
    ${CMAKE_BINARY_DIR}/tinyopt.js
  COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_BINARY_DIR}/tinyopt.wasm
    ${CMAKE_BINARY_DIR}/tinyopt.wasm
  COMMENT "Copying WASM module to build root for testing"
)
