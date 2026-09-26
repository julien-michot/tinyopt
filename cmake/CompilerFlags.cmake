# 1. Use standard variables for global behavior, but prefer target-based for libraries
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# If cross-compiling to Emscripten, ensure macOS-specific linker flags are not
# propagated to the em++/wasm-ld command line, which will reject them.
if (CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    if(DEFINED CMAKE_SHARED_LINKER_FLAGS)
        string(REPLACE "-Wl,-headerpad_max_install_names" "" CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}")
        string(REPLACE "-Wl,-dead_strip_dylibs" "" CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}")
        string(REPLACE "-Wl,-rpath" "" CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}")
    endif()
    if(DEFINED CMAKE_EXE_LINKER_FLAGS)
        string(REPLACE "-Wl,-headerpad_max_install_names" "" CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
        string(REPLACE "-Wl,-dead_strip_dylibs" "" CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
        string(REPLACE "-Wl,-rpath" "" CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
    endif()
endif()


# On macOS, sanitize any environment-injected architecture tuning flags that AppleClang
# doesn't accept (e.g. -march=core2, -mtune=haswell). These can leak from conda
# or other toolchains into CMAKE_C_FLAGS_INIT and cause the initial compiler check
# in project() to fail. We remove common problematic patterns before calling project().
if(APPLE)
  if(DEFINED CMAKE_C_FLAGS_INIT)
    string(REGEX REPLACE "-march=[^ ]*" "" CMAKE_C_FLAGS_INIT "${CMAKE_C_FLAGS_INIT}")
    string(REGEX REPLACE "-mtune=[^ ]*" "" CMAKE_C_FLAGS_INIT "${CMAKE_C_FLAGS_INIT}")
    string(REGEX REPLACE "-mssse3" "" CMAKE_C_FLAGS_INIT "${CMAKE_C_FLAGS_INIT}")
  endif()
  if(DEFINED CMAKE_CXX_FLAGS_INIT)
    string(REGEX REPLACE "-march=[^ ]*" "" CMAKE_CXX_FLAGS_INIT "${CMAKE_CXX_FLAGS_INIT}")
    string(REGEX REPLACE "-mtune=[^ ]*" "" CMAKE_CXX_FLAGS_INIT "${CMAKE_CXX_FLAGS_INIT}")
    string(REGEX REPLACE "-mssse3" "" CMAKE_CXX_FLAGS_INIT "${CMAKE_CXX_FLAGS_INIT}")
  endif()
  # Also clear common environment variables that may inject incompatible flags
  # (for example conda/pixi environments export CFLAGS/CXXFLAGS). Clearing them
  # here prevents those flags from being appended to try-compile commands.
  set(ENV{CFLAGS} "")
  set(ENV{CXXFLAGS} "")
  set(ENV{CPPFLAGS} "")
  set(CMAKE_C_FLAGS_INIT "")
  set(CMAKE_CXX_FLAGS_INIT "")
  # Fix for Conda/Pixi + C++20 availability macro errors
  add_compile_definitions(_LIBCPP_DISABLE_AVAILABILITY)
endif()

option(TINYOPT_ENABLE_NATIVE_TUNING "Enable native CPU architecture tuning" ON)

# 2. Add ASAN configuration properly
# Note: This only works if done before the project() call or very early.
get_property(is_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(is_multi_config)
    if(NOT "ASAN" IN_LIST CMAKE_CONFIGURATION_TYPES)
        list(APPEND CMAKE_CONFIGURATION_TYPES ASAN)
    endif()
endif()

function(tinyopt_set_compiler_flags)
    set(options)
    set(oneValueArgs TARGET SCOPE)
    set(multiValueArgs)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT ARG_TARGET)
        message(FATAL_ERROR "tinyopt_set_compiler_flags: TARGET argument is required.")
    endif()

    set(scope ${ARG_SCOPE})
    if(NOT scope)
        set(scope "PRIVATE")
    endif()

    # --- Modern PIC Handling ---
    set_target_properties(${ARG_TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)

    # --- Warnings & Diagnostics ---
    # Using 'target_compile_options' with platform checks
    if (MSVC)
        target_compile_options(${ARG_TARGET} ${scope} /W3 /wd5054)
    else()
        target_compile_options(${ARG_TARGET} ${scope} -Wall -Wextra -Werror)

        # Color diagnostics (Cleaner logic)
        set(color_flag $<$<CXX_COMPILER_ID:Clang>:-fcolor-diagnostics> $<$<CXX_COMPILER_ID:GNU>:-fdiagnostics-color=always>)
        target_compile_options(${ARG_TARGET} ${scope} ${color_flag})
    endif()

    # --- Architecture Tuning ---
    # Only enable -march=native for compilers that support it (GNU and non-Apple Clang).
    # AppleClang doesn't accept -march and will fail configuration on macOS.
    # Avoid adding -march when cross-compiling to Emscripten or when the
    # compiler/toolchain doesn't support it. Emscripten's clang rejects
    # -march= and will fail the build. We check the system name to detect
    # Emscripten targets and skip native tuning in that case.
    if (TINYOPT_ENABLE_NATIVE_TUNING AND NOT MSVC AND NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
        if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
            target_compile_options(${ARG_TARGET} ${scope} "-march=native")
        else()
            # Skip -march for AppleClang and other toolchains that don't support it.
        endif()
    endif()

    # --- Sanitizer Flags ---
    # We avoid manually setting -O3/-O2 because CMAKE_BUILD_TYPE handles that.
    # We only add the specific sanitizer overhead here.
    set(ASAN_COMPILE_FLAGS
        -fsanitize=address
        -fsanitize-address-use-after-scope
        -fno-optimize-sibling-calls
        -fno-omit-frame-pointer
    )

    target_compile_options(${ARG_TARGET} ${scope} $<$<CONFIG:ASAN>:${ASAN_COMPILE_FLAGS}>)
    target_link_options(${ARG_TARGET} ${scope} $<$<CONFIG:ASAN>:-fsanitize=address>)

    # --- CUDA Support ---
    # Check if CUDA is enabled as a language globally
    get_property(enabled_languages GLOBAL PROPERTY ENABLED_LANGUAGES)
    if("CUDA" IN_LIST enabled_languages)
        # Modern way: Use the CMAKE_CUDA_COMPILER_ID generator expression
        target_compile_options(${ARG_TARGET} ${scope}
            $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:NVIDIA>>:-Xcompiler=-fPIC>
        )
    endif()

endfunction()