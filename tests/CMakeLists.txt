
add_executable(tinyopt_test_simple simple.cpp)
add_test(NAME tinyopt_test_simple COMMAND tinyopt_test_simple)
target_link_libraries(tinyopt_test_simple PRIVATE ${THIRDPARTY_TEST_LIBS} tinyopt)

add_executable(tinyopt_test_types types.cpp)
add_test(NAME tinyopt_test_types COMMAND tinyopt_test_types)
target_link_libraries(tinyopt_test_types PRIVATE ${THIRDPARTY_TEST_LIBS} tinyopt)

add_executable(tinyopt_test_sqrt2 sqrt2.cpp)
add_test(NAME tinyopt_test_sqrt2 COMMAND tinyopt_test_sqrt2)
target_link_libraries(tinyopt_test_sqrt2 PRIVATE ${THIRDPARTY_TEST_LIBS} tinyopt)

add_executable(tinyopt_test_fit_circle circle.cpp)
add_test(NAME tinyopt_test_fit_circle COMMAND tinyopt_test_fit_circle)
target_link_libraries(tinyopt_test_fit_circle PRIVATE ${THIRDPARTY_TEST_LIBS} tinyopt)

add_executable(tinyopt_test_userdef_params userdef_params.cpp)
add_test(NAME tinyopt_test_userdef_params COMMAND tinyopt_test_userdef_params)
target_link_libraries(tinyopt_test_userdef_params PRIVATE ${THIRDPARTY_TEST_LIBS} tinyopt)

add_executable(tinyopt_test_userdef_params_jet userdef_params_jet.cpp)
add_test(NAME tinyopt_test_userdef_params_jet COMMAND tinyopt_test_userdef_params_jet)
target_link_libraries(tinyopt_test_userdef_params_jet PRIVATE ${THIRDPARTY_TEST_LIBS} tinyopt)

if (BUILD_TINYOPT_SOPHUS_EXAMPLES)
  add_executable(tinyopt_test_sophus sophus.cpp)
  add_test(NAME tinyopt_test_sophus COMMAND tinyopt_test_sophus)
  target_link_libraries(tinyopt_test_sophus PRIVATE ${THIRDPARTY_TEST_LIBS} tinyopt)
endif(BUILD_TINYOPT_SOPHUS_EXAMPLES)

if (BUILD_TINYOPT_LIEPLUSPLUS_EXAMPLES)
  add_executable(tinyopt_test_lie++ lieplusplus.cpp)
  add_test(NAME tinyopt_test_lie++ COMMAND tinyopt_test_lie++)
  target_link_libraries(tinyopt_test_lie++ PRIVATE ${THIRDPARTY_TEST_LIBS} tinyopt)
endif(BUILD_TINYOPT_LIEPLUSPLUS_EXAMPLES)