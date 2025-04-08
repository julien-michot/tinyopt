
include(Benchmarking)

add_executable(tinyopt_bench_densef dense_float.cpp)
target_link_libraries(tinyopt_bench_densef PRIVATE ${THIRDPARTY_TEST_LIBS} tinyopt)
add_benchmark_target(tinyopt_bench_densef)

add_executable(tinyopt_bench_dense dense_double.cpp)
target_link_libraries(tinyopt_bench_dense PRIVATE ${THIRDPARTY_TEST_LIBS} tinyopt)
add_benchmark_target(tinyopt_bench_dense)

add_executable(tinyopt_bench_sparse sparse.cpp)
target_link_libraries(tinyopt_bench_sparse PRIVATE ${THIRDPARTY_TEST_LIBS} tinyopt)
add_benchmark_target(tinyopt_bench_sparse)

# Benchmark executables
set(BENCHMARK_EXECUTABLES tinyopt_bench_densef tinyopt_bench_dense tinyopt_bench_sparse)

# Make bench
#TODO add_benchmark_all_targets(bench "${BENCHMARK_EXECUTABLES}")
#TODO export to a benchmarks.json file