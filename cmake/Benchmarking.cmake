# Benchmarking rules

# Define a custom target to run benchmarks for a specific executable
function(add_benchmark_target executable_name)
  target_link_libraries(${executable_name} PRIVATE tinyopt Catch2::Catch2WithMain)
  set_target_properties(${executable_name} PROPERTIES LABELS "Benchmark Tinyopt")
  add_custom_target(run_${executable_name}
    COMMAND $<TARGET_FILE:${executable_name}> "[benchmark]" --benchmark-no-analysis
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running benchmarks in ${executable_name}..."
    VERBATIM
  )
endfunction()
