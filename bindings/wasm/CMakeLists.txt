# The module is the C library itself: no wrapper source, JavaScript drives the C API directly.
set(TINYOPT_WASM_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(TINYOPT_WASM_OPTIMIZER "${TINYOPT_WASM_OUT_DIR}/optimizer.js")
set(TINYOPT_WASM_EXPORTS "_tinyopt_optimize,_tinyopt_options_default,_malloc,_free")
set(TINYOPT_WASM_RUNTIME
    "addFunction,removeFunction,HEAPF64,HEAPF32,HEAP32,HEAPU32,HEAP16,HEAPU16,HEAP8,HEAPU8")

# Pixi injects a native stack-protector flag that Emscripten cannot link.
target_compile_options(tinyopt_c PRIVATE -fno-stack-protector)

find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_custom_command(
  OUTPUT "${TINYOPT_WASM_OPTIMIZER}"
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/generate_optimizer.py"
          --template "${CMAKE_SOURCE_DIR}/examples/wasm/optimizer.js"
          --common-header "${CMAKE_SOURCE_DIR}/include/tinyopt/c/c_api_common.h"
          --double-header "${CMAKE_SOURCE_DIR}/include/tinyopt/c/c_api_double.h"
          --output "${TINYOPT_WASM_OPTIMIZER}"
  DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/generate_optimizer.py"
    "${CMAKE_SOURCE_DIR}/examples/wasm/optimizer.js"
    "${CMAKE_SOURCE_DIR}/include/tinyopt/c/c_api_common.h"
    "${CMAKE_SOURCE_DIR}/include/tinyopt/c/c_api_double.h"
  COMMENT "Generating the WebAssembly JavaScript optimizer binding"
  VERBATIM)

add_custom_command(
  OUTPUT "${TINYOPT_WASM_OUT_DIR}/tinyopt.mjs" "${TINYOPT_WASM_OUT_DIR}/tinyopt.wasm"
  COMMAND ${CMAKE_CXX_COMPILER} -O3 -fexceptions --no-entry
          $<TARGET_FILE:tinyopt_c>
          -o "${TINYOPT_WASM_OUT_DIR}/tinyopt.mjs"
          -sMODULARIZE=1 -sEXPORT_ES6=1 -sEXPORT_NAME=createTinyopt
          -sENVIRONMENT=web,node
          -sALLOW_MEMORY_GROWTH=1 -sALLOW_TABLE_GROWTH=1
          -sEXPORTED_FUNCTIONS=${TINYOPT_WASM_EXPORTS}
          -sEXPORTED_RUNTIME_METHODS=${TINYOPT_WASM_RUNTIME}
  DEPENDS tinyopt_c
  COMMENT "Linking tinyopt.mjs / tinyopt.wasm"
  VERBATIM)
add_custom_target(tinyopt_wasm
  DEPENDS "${TINYOPT_WASM_OUT_DIR}/tinyopt.mjs" "${TINYOPT_WASM_OPTIMIZER}")

# Bundle the module with the JS example so `build-wasm/wasm-example` can be served as is.
set(TINYOPT_WASM_EXAMPLE_DIR "${CMAKE_BINARY_DIR}/wasm-example")
add_custom_target(tinyopt_wasm_example ALL
  COMMAND ${CMAKE_COMMAND} -E make_directory "${TINYOPT_WASM_EXAMPLE_DIR}"
  COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_SOURCE_DIR}/examples/wasm"
          "${TINYOPT_WASM_EXAMPLE_DIR}"
  COMMAND ${CMAKE_COMMAND} -E copy
          "${TINYOPT_WASM_OUT_DIR}/tinyopt.mjs" "${TINYOPT_WASM_OUT_DIR}/tinyopt.wasm"
          "${TINYOPT_WASM_EXAMPLE_DIR}"
  COMMAND ${CMAKE_COMMAND} -E copy
          "${TINYOPT_WASM_OPTIMIZER}" "${TINYOPT_WASM_EXAMPLE_DIR}/optimizer.js"
  DEPENDS tinyopt_wasm
  COMMENT "Assembling the WebAssembly example in ${TINYOPT_WASM_EXAMPLE_DIR}"
  VERBATIM)
