
include(Benchmarking)

add_executable(tinyopt_bench_tinyopt dense.cpp densef.cpp sparse.cpp)
add_benchmark_target(tinyopt_bench_tinyopt)

# Sparse only
add_executable(tinyopt_bench_sparse sparse.cpp)
add_benchmark_target(tinyopt_bench_sparse)

# Dense only
add_executable(tinyopt_bench_dense dense.cpp)
add_benchmark_target(tinyopt_bench_dense)


# Benchmark executables
set(BENCHMARK_EXECUTABLES tinyopt_bench_tinyopt)

if (TINYOPT_BUILD_CERES)
  add_executable(tinyopt_bench_ceres ceres.cpp)
  add_benchmark_target(tinyopt_bench_ceres)
  list(APPEND BENCHMARK_EXECUTABLES tinyopt_bench_ceres)
endif()

# Create a custom target to run all benchmarks
set(RUN_COMMANDS "")
foreach(tgt IN LISTS BENCHMARK_EXECUTABLES)
    # Append the word 'COMMAND' followed by the executable path
    list(APPEND RUN_COMMANDS COMMAND "$<TARGET_FILE:${tgt}>")
endforeach()

add_custom_target(run_all_benchmarks
    ${RUN_COMMANDS}
    COMMENT "Executing all benchmarks..."
    VERBATIM
)