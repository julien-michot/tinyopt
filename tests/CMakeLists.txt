
add_executable(tinyopt_test_sqrt2 sqrt2.cpp)
add_test(NAME tinyopt_test_sqrt2 COMMAND tinyopt_test_sqrt2)
target_link_libraries(tinyopt_test_sqrt2 PRIVATE ${THIRDPARTY_TEST_LIBS})

add_executable(tinyopt_test_fit_circle circle.cpp)
add_test(NAME tinyopt_test_fit_circle COMMAND tinyopt_test_fit_circle)
target_link_libraries(tinyopt_test_fit_circle PRIVATE ${THIRDPARTY_TEST_LIBS})
