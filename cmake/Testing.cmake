
# Tests
enable_testing()

# Define a custom target to run tests
function(add_test_target test_name)
    add_test(NAME $<TARGET_FILE:${test_name}> COMMAND ${test_name})
    set_property(TEST $<TARGET_FILE:${test_name}> PROPERTY LABELS TESTLABEL Tinyopt)
    add_custom_target(run_${test_name}
        COMMAND $<TARGET_FILE:${test_name}> "" --benchmark-no-analysis
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Running tests in ${test_name}..."
        VERBATIM
    )
endfunction()
