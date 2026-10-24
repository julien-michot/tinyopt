# Copyright 2026 Julien Michot.
# SPDX-License-Identifier: Apache-2.0

function(add_ceres_exact_test test_name source)
  add_executable(${test_name} ${source})
  target_link_libraries(${test_name} PRIVATE tinyopt Catch2::Catch2WithMain Ceres::ceres)
  target_include_directories(${test_name} PRIVATE ${CMAKE_SOURCE_DIR}/benchmarks/common)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${test_name} PRIVATE -O3)
  elseif(MSVC)
    target_compile_options(${test_name} PRIVATE /O2)
  endif()
  add_test(NAME ${test_name} COMMAND ${test_name})
  set_property(TEST ${test_name} PROPERTY LABELS Tinyopt CeresExact)
endfunction()

add_ceres_exact_test(tinyopt_test_exact_lm test_lm.cpp)
add_ceres_exact_test(tinyopt_test_exact_dogleg test_dogleg.cpp)
add_ceres_exact_test(tinyopt_test_exact_gn test_gn.cpp)
add_ceres_exact_test(tinyopt_test_exact_bfgs test_bfgs.cpp)
add_ceres_exact_test(tinyopt_test_exact_lbfgs test_lbfgs.cpp)
add_ceres_exact_test(tinyopt_test_exact_cg test_cg.cpp)
add_ceres_exact_test(tinyopt_test_exact_gd test_gd.cpp)
add_ceres_exact_test(tinyopt_test_exact_ba_block_sparse test_ba_block_sparse.cpp)

add_custom_target(tinyopt_tests_exact
  DEPENDS
    tinyopt_test_exact_lm
    tinyopt_test_exact_dogleg
    tinyopt_test_exact_gn
    tinyopt_test_exact_bfgs
    tinyopt_test_exact_lbfgs
    tinyopt_test_exact_cg
    tinyopt_test_exact_gd
    tinyopt_test_exact_ba_block_sparse
)
