set(CMAKE_CXX_STANDARD 20) # minimum c++ version (C++ 17 works but has a dummy std::format)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
message(STATUS "Tinyopt: using C++ standard: ${CMAKE_CXX_STANDARD}")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # generate compile_commands.json for clangd

# Check the C++ compiler, define flags
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  message(STATUS "Using GCC C++ compiler")
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:-fPIC>)
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:-Wall>)
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:-Wextra>)
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:-Werror>)
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:-fdiagnostics-color=always>)
  # TODO -Wconversion
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  message(STATUS "Using Clang C++ compiler")
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:-fPIC>)
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:-Wall>)
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:-Wextra>)
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:-Werror>)
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:-fcolor-diagnostics>)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  message(STATUS "Using MSVC C++ compiler")
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/W3>)
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/wd5054>) # 5054 is for Eigen
  # TODO /WX
else()
  message(STATUS "Unknown C++ compiler: ${CMAKE_CXX_COMPILER_ID}")
endif()

# ASAN with ConfigType Address Sanitizer (use -DCMAKE_BUILD_TYPE=ASAN)
list(APPEND CMAKE_CONFIGURATION_TYPES ASAN)

if(CMAKE_BUILD_TYPE MATCHES "ASAN")
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:-fsanitize=address>)
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:-fsanitize-address-use-after-scope>)
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:-fno-optimize-sibling-calls>)
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:-fno-omit-frame-pointer>)
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:-O1>)
  link_libraries("-fsanitize=address")
elseif(CMAKE_BUILD_TYPE MATCHES "RelWithDebInfo")
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:-O2>)
elseif(CMAKE_BUILD_TYPE MATCHES "Release")
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:-O3>)
endif()
