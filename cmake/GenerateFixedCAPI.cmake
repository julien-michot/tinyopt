set(TINYOPT_C_GENERATED_ROOT "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(TINYOPT_C_FIXED_INCLUDE_DIR "${TINYOPT_C_GENERATED_ROOT}/include")
set(TINYOPT_C_FIXED_SOURCE_DIR "${TINYOPT_C_GENERATED_ROOT}/src")
set(TINYOPT_C_FIXED_TEST_DIR "${TINYOPT_C_GENERATED_ROOT}/tests/c")
file(REMOVE_RECURSE "${TINYOPT_C_GENERATED_ROOT}")
file(MAKE_DIRECTORY
  "${TINYOPT_C_FIXED_INCLUDE_DIR}/tinyopt/c"
  "${TINYOPT_C_FIXED_SOURCE_DIR}"
  "${TINYOPT_C_FIXED_TEST_DIR}")

if(TINYOPT_C_API_FLOAT)
  set(TINYOPT_C_API_FLOAT_VALUE 1)
else()
  set(TINYOPT_C_API_FLOAT_VALUE 0)
endif()
if(TINYOPT_BUILD_SHARED_C)
  set(TINYOPT_C_API_SHARED_VALUE 1)
else()
  set(TINYOPT_C_API_SHARED_VALUE 0)
endif()
if(TINYOPT_ENABLE_SUITESPARSE)
  set(TINYOPT_C_API_SUITESPARSE_VALUE 1)
else()
  set(TINYOPT_C_API_SUITESPARSE_VALUE 0)
endif()
file(WRITE "${TINYOPT_C_FIXED_INCLUDE_DIR}/tinyopt/c/c_api_config.h"
  "// Copyright 2026 Julien Michot.\n// SPDX-License-Identifier: Apache-2.0\n\n"
  "#ifndef TINYOPT_C_C_API_CONFIG_H\n#define TINYOPT_C_C_API_CONFIG_H\n"
  "#define TINYOPT_C_API_ENABLE_FLOAT ${TINYOPT_C_API_FLOAT_VALUE}\n"
  "#define TINYOPT_C_API_ENABLE_SUITESPARSE ${TINYOPT_C_API_SUITESPARSE_VALUE}\n"
  "#define TINYOPT_C_API_ENABLE_SHARED ${TINYOPT_C_API_SHARED_VALUE}\n#endif\n")

set(TINYOPT_C_GENERATED_WRAPPERS "")
set(TINYOPT_C_GENERATED_BACKENDS "")
set(_tinyopt_fixed_includes "")
set(_tinyopt_float_backend "")
set(_tinyopt_double_backend "")
set(_tinyopt_float_api FALSE)
set(_tinyopt_double_api FALSE)

set(_tinyopt_supported_fixed_sizes 1 2 3 4 5 6 10 12)
list(REMOVE_DUPLICATES TINYOPT_C_FIXED_SIZES)
foreach(_dimension IN LISTS TINYOPT_C_FIXED_SIZES)
  if(NOT _dimension IN_LIST _tinyopt_supported_fixed_sizes)
    message(FATAL_ERROR
      "Unsupported TINYOPT_C_FIXED_SIZES entry '${_dimension}'. Supported values: 1, 2, 3, 4, 5, 6, 10, 12")
  endif()

  set(_fixed_suffixes d)
  if(TINYOPT_C_API_FLOAT)
    list(APPEND _fixed_suffixes f)
  endif()
  foreach(_suffix IN LISTS _fixed_suffixes)
    if(_suffix STREQUAL "f")
      set(SCALAR_TYPE float)
      set(LONG_SUFFIX float)
      set(_backend_variable _tinyopt_float_backend)
      set(_problem_type tinyopt_problemf_t)
      set(SUFFIX f)
      set(INTERNAL_SUFFIX f)
      set(_tinyopt_float_api TRUE)
    else()
      set(SCALAR_TYPE double)
      set(LONG_SUFFIX double)
      set(_backend_variable _tinyopt_double_backend)
      set(_problem_type tinyopt_problem_t)
      set(SUFFIX "")
      set(INTERNAL_SUFFIX d)
      set(_tinyopt_double_api TRUE)
    endif()
    set(DIMENSION "${_dimension}")
    set(PROBLEM_TYPE "${_problem_type}")
    string(TOUPPER "${_dimension}_${_suffix}" _guard_suffix)
    set(HEADER_GUARD "TINYOPT_C_API_FIXED_${_guard_suffix}_H")

    set(_fixed_header
      "${TINYOPT_C_FIXED_INCLUDE_DIR}/tinyopt/c/c_api_fixed_${_dimension}${SUFFIX}.h")
    set(_fixed_wrapper
      "${TINYOPT_C_FIXED_SOURCE_DIR}/c_api_fixed_${_dimension}_${_suffix}.c")
    configure_file("${PROJECT_SOURCE_DIR}/cmake/templates/c_api_fixed.h.in"
                   "${_fixed_header}" @ONLY)
    configure_file("${PROJECT_SOURCE_DIR}/cmake/templates/c_api_fixed.c.in"
                   "${_fixed_wrapper}" @ONLY)
    list(APPEND TINYOPT_C_GENERATED_WRAPPERS "${_fixed_wrapper}")
    string(APPEND _tinyopt_fixed_includes
      "#include <tinyopt/c/c_api_fixed_${_dimension}${SUFFIX}.h>\n")

    string(APPEND ${_backend_variable}
      "extern \"C\" tinyopt_status_t tinyopt_optimize_fixed_${_suffix}_${_dimension}(\n"
      "    ${SCALAR_TYPE} *x, void (*plus_eq)(${SCALAR_TYPE} *, ${SCALAR_TYPE} *),\n"
      "    const ${_problem_type} *problem, const tinyopt_options_t *options, tinyopt_summary_t *summary) {\n"
      "  return tinyopt::c_api_detail::OptimizeFixed<${SCALAR_TYPE}, ${_dimension}>(\n"
      "      x, plus_eq, problem, options, summary);\n}\n\n")

    if(TINYOPT_BUILD_TESTS)
      set(_test_source
        "${TINYOPT_C_FIXED_TEST_DIR}/c_api_fixed_test_${_dimension}_${_suffix}.c")
      configure_file("${PROJECT_SOURCE_DIR}/tests/c/templates/c_api_fixed_test.c.in"
                     "${_test_source}" @ONLY)
    endif()
  endforeach()
endforeach()

file(WRITE "${TINYOPT_C_FIXED_INCLUDE_DIR}/tinyopt/c/c_api_fixed.h"
  "// Copyright 2026 Julien Michot.\n// SPDX-License-Identifier: Apache-2.0\n\n"
  "#ifndef TINYOPT_C_C_API_FIXED_H\n#define TINYOPT_C_C_API_FIXED_H\n\n"
  "${_tinyopt_fixed_includes}\n#endif\n")

set(_tinyopt_float_includes "")
set(_tinyopt_double_includes "")
foreach(_dimension IN LISTS TINYOPT_C_FIXED_SIZES)
  if(TINYOPT_C_API_FLOAT)
    string(APPEND _tinyopt_float_includes
      "#include <tinyopt/c/c_api_fixed_${_dimension}f.h>\n")
  endif()
  string(APPEND _tinyopt_double_includes
    "#include <tinyopt/c/c_api_fixed_${_dimension}.h>\n")
endforeach()
file(WRITE "${TINYOPT_C_FIXED_INCLUDE_DIR}/tinyopt/c/c_api_fixed_float.h"
  "#ifndef TINYOPT_C_C_API_FIXED_FLOAT_H\n#define TINYOPT_C_C_API_FIXED_FLOAT_H\n\n"
  "${_tinyopt_float_includes}\n#endif\n")
file(WRITE "${TINYOPT_C_FIXED_INCLUDE_DIR}/tinyopt/c/c_api_fixed_double.h"
  "#ifndef TINYOPT_C_C_API_FIXED_DOUBLE_H\n#define TINYOPT_C_C_API_FIXED_DOUBLE_H\n\n"
  "${_tinyopt_double_includes}\n#endif\n")

if(_tinyopt_float_api)
  set(BACKEND_DEFINITIONS "${_tinyopt_float_backend}")
  set(_backend_file "${TINYOPT_C_FIXED_SOURCE_DIR}/c_api_fixed_float.cpp")
  configure_file("${PROJECT_SOURCE_DIR}/cmake/templates/c_api_fixed_backend.cpp.in"
                 "${_backend_file}" @ONLY)
  list(APPEND TINYOPT_C_GENERATED_BACKENDS "${_backend_file}")
endif()
if(_tinyopt_double_api)
  set(BACKEND_DEFINITIONS "${_tinyopt_double_backend}")
  set(_backend_file "${TINYOPT_C_FIXED_SOURCE_DIR}/c_api_fixed_double.cpp")
  configure_file("${PROJECT_SOURCE_DIR}/cmake/templates/c_api_fixed_backend.cpp.in"
                 "${_backend_file}" @ONLY)
  list(APPEND TINYOPT_C_GENERATED_BACKENDS "${_backend_file}")
endif()