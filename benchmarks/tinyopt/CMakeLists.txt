add_executable(tinyopt_bench_tinyopt dense.cpp densef.cpp sparse.cpp robust_pose.cpp
  bundle_adjustment.cpp)
add_benchmark_target(tinyopt_bench_tinyopt)
# Sophus is only used by the pose benchmarks and is not part of the tinyopt interface.
if(TARGET Sophus::Sophus)
  target_link_libraries(tinyopt_bench_tinyopt PRIVATE Sophus::Sophus)
elseif(Sophus_INCLUDE_DIR)
  target_include_directories(tinyopt_bench_tinyopt PRIVATE ${Sophus_INCLUDE_DIR})
endif()

add_executable(tinyopt_bench_sparse sparse.cpp)
add_benchmark_target(tinyopt_bench_sparse)

add_executable(tinyopt_bench_dense dense.cpp)
add_benchmark_target(tinyopt_bench_dense)

add_subdirectory(profiling)

if (TINYOPT_ENABLE_SUITESPARSE)
  add_executable(tinyopt_bench_suitesparse suitesparse.cpp)
  add_benchmark_target(tinyopt_bench_suitesparse)
endif()