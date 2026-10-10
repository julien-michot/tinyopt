
include(Benchmarking)

add_executable(tinyopt_bench_simple_optimize EXCLUDE_FROM_ALL tinyopt/simple_optimize.cpp)
target_link_libraries(tinyopt_bench_simple_optimize PRIVATE tinyopt)

add_subdirectory(tinyopt)

if (TINYOPT_BUILD_CERES)
  add_subdirectory(ceres)
endif()

if (TINYOPT_BUILD_G2O_BENCHMARKS)
  add_subdirectory(g2o)
endif()

if (TINYOPT_BUILD_GTSAM_BENCHMARKS)
  add_subdirectory(gtsam)
endif()

set(BENCHMARK_EXECUTABLES tinyopt_bench_tinyopt)
foreach(backend IN ITEMS ceres g2o gtsam)
  if (TARGET tinyopt_bench_${backend})
    list(APPEND BENCHMARK_EXECUTABLES tinyopt_bench_${backend})
  endif()
endforeach()
if (TARGET tinyopt_bench_ceres_tinysolver)
  list(APPEND BENCHMARK_EXECUTABLES tinyopt_bench_ceres_tinysolver)
endif()

set(RUN_COMMANDS "")
foreach(target IN LISTS BENCHMARK_EXECUTABLES)
  list(APPEND RUN_COMMANDS COMMAND "$<TARGET_FILE:${target}>" "[benchmark]"
    --benchmark-no-analysis)
endforeach()

add_custom_target(run_all_benchmarks
  ${RUN_COMMANDS}
  COMMENT "Executing all available benchmarks..."
  VERBATIM
)