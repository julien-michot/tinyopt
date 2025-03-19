
add_executable(tinyopt_test_sqrt2 sqrt2.cpp)
add_test(NAME tinyopt_test_sqrt2 COMMAND tinyopt_test_sqrt2)
target_link_libraries(tinyopt_test_sqrt2 PRIVATE ${THIRDPARTY_TEST_LIBS})
