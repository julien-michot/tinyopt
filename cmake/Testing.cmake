
# Tests
enable_testing()

# Define a custom target to run tests
function(add_test_target test_name)
    target_link_libraries(${test_name} PRIVATE tinyopt Catch2::Catch2WithMain)

    add_test(NAME ${test_name} COMMAND ${test_name})
    set_property(TEST ${test_name} PROPERTY LABELS Tinyopt Tests)

endfunction()
