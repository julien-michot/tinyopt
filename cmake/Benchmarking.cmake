# Benchmarking rules

# Define a custom target to run benchmarks for a specific executable
function(add_benchmark_target executable_name)
  add_custom_target(run_${executable_name}
    COMMAND $<TARGET_FILE:${executable_name}> "[benchmark]" --benchmark-no-analysis
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running benchmarks in ${executable_name}..."
    VERBATIM
  )
endfunction()

# Create a custom target for each benchmark executables
function(add_benchmark_all_targets target_name executable_list)
  add_custom_target(${target_name}
    COMMAND
    ${CMAKE_COMMAND} -E echo "Running benchmarks..."
    COMMAND
    foreach (exe ${executable_list})
      ${CMAKE_COMMAND} -E echo "Running: $<TARGET_FILE:${exe}>"
      COMMAND
      $<TARGET_FILE:${exe}>
    endforeach ()
    DEPENDS ${executable_list}
    COMMENT "Runs all benchmark executables"
  )
endfunction()
