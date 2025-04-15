set(CMAKE_CXX_STANDARD 20) # minimum c++ version (C++ 17 works but has a dummy std::format)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
message(STATUS "Tinyopt: using C++ standard: ${CMAKE_CXX_STANDARD}")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # generate compile_commands.json for clangd

# Check the C++ compiler, define flags
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  message(STATUS "Using GCC C++ compiler")
  set(COMPILER_FLAGS -fPIC -Wall -Wextra -Werror -fdiagnostics-color=always ) # -Wconversion
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  message(STATUS "Using Clang C++ compiler")
  set(COMPILER_FLAGS -fPIC -Wall -Wextra -Werror -fcolor-diagnostics)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  message(STATUS "Using MSVC C++ compiler")
  set(COMPILER_FLAGS /W3 /wd5054) # 5054 is for Eigen,  /WX
else()
  message(STATUS "Unknown C++ compiler: ${CMAKE_CXX_COMPILER_ID}")
endif()
# Add the compiler flags
add_compile_options(${COMPILER_FLAGS})

# ASAN with ConfigType Address Sanitizer (use -DCMAKE_BUILD_TYPE=ASAN)
list(APPEND CMAKE_CONFIGURATION_TYPES ASAN)

if(CMAKE_BUILD_TYPE MATCHES "ASAN")
  add_compile_options("-fsanitize=address")
  add_compile_options("-fsanitize-address-use-after-scope")
  add_compile_options("-fno-optimize-sibling-calls")
  add_compile_options("-fno-omit-frame-pointer")
  add_compile_options("-O1")
  link_libraries("-fsanitize=address")
elseif(CMAKE_BUILD_TYPE MATCHES "RelWithDebInfo")
  add_compile_options("-O2")
elseif(CMAKE_BUILD_TYPE MATCHES "Release")
  add_compile_options("-O3")
endif()
