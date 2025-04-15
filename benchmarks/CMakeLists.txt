
include(Benchmarking)

add_executable(tinyopt_bench_tinyopt dense.cpp densef.cpp sparse.cpp)
target_link_libraries(tinyopt_bench_tinyopt PRIVATE ${THIRDPARTY_TEST_LIBS} tinyopt)
add_benchmark_target(tinyopt_bench_tinyopt)

# Sparse only
add_executable(tinyopt_bench_sparse sparse.cpp)
target_link_libraries(tinyopt_bench_sparse PRIVATE ${THIRDPARTY_TEST_LIBS} tinyopt)
add_benchmark_target(tinyopt_bench_sparse)

# Dense only
add_executable(tinyopt_bench_dense dense.cpp)
target_link_libraries(tinyopt_bench_dense PRIVATE ${THIRDPARTY_TEST_LIBS} tinyopt)
add_benchmark_target(tinyopt_bench_dense)


# Benchmark executables
set(BENCHMARK_EXECUTABLES tinyopt_bench_tinyopt)

if (TINYOPT_BUILD_CERES)
  add_executable(tinyopt_bench_ceres ceres.cpp)
  target_link_libraries(tinyopt_bench_ceres PRIVATE ${THIRDPARTY_TEST_LIBS} tinyopt)
  add_benchmark_target(tinyopt_bench_ceres)
  set(BENCHMARK_EXECUTABLES ${BENCHMARK_EXECUTABLES} tinyopt_bench_ceres)
endif()


# Make bench
#TODO add_benchmark_all_targets(bench "${BENCHMARK_EXECUTABLES}")
#TODO export to a benchmarks.json file