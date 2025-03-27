set (CMAKE_CXX_STANDARD 20)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # generate compile_commands.json for clangd

if (MSVC)
  set(COMPILER_FLAGS /W4 /WX)
else ()
  set(COMPILER_FLAGS
      -std=c++20 # cmake bug?
      -fPIC -Wall -Wextra -pedantic -Werror
      -fdiagnostics-color=always
      -Wno-language-extension-token
      -Wno-gnu-statement-expression # statement as expression
      )
endif ()
add_compile_options(${COMPILER_FLAGS})

# ASAN with ConfigType Address Sanitizer (use -DCMAKE_BUILD_TYPE=ASAN)
list(APPEND CMAKE_CONFIGURATION_TYPES ASAN)

if (CMAKE_BUILD_TYPE MATCHES "ASAN")
  add_compile_options("-fsanitize=address")
  add_compile_options("-fsanitize-address-use-after-scope")
  add_compile_options("-fno-optimize-sibling-calls")
  add_compile_options("-fno-omit-frame-pointer")
  add_compile_options("-O1")
  link_libraries("-fsanitize=address")
elseif (CMAKE_BUILD_TYPE MATCHES "RelWithDebInfo")
  add_compile_options("-O2")
elseif (CMAKE_BUILD_TYPE MATCHES "Release")
  add_compile_options("-O3")
endif ()

# Options
if (NOT ENABLE_FORMATTERS)
  add_definitions(-DTINYOPT_NO_FORMATTERS=1)
endif ()
