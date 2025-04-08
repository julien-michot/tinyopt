
add_executable(tinyopt_test_loss loss.cpp)
target_link_libraries(tinyopt_test_loss PRIVATE ${THIRDPARTY_TEST_LIBS} tinyopt)
add_test_target(tinyopt_test_loss)

add_executable(tinyopt_test_diff diff.cpp)
target_link_libraries(tinyopt_test_diff PRIVATE ${THIRDPARTY_TEST_LIBS} tinyopt)
add_test_target(tinyopt_test_diff)

add_executable(tinyopt_test_basic basic.cpp)
target_link_libraries(tinyopt_test_basic PRIVATE ${THIRDPARTY_TEST_LIBS} tinyopt)
add_test_target(tinyopt_test_basic)

add_executable(tinyopt_test_simple simple.cpp)
target_link_libraries(tinyopt_test_simple PRIVATE ${THIRDPARTY_TEST_LIBS} tinyopt)
add_test_target(tinyopt_test_simple)

add_executable(tinyopt_test_types types.cpp)
target_link_libraries(tinyopt_test_types PRIVATE ${THIRDPARTY_TEST_LIBS} tinyopt)
add_test_target(tinyopt_test_types)

add_executable(tinyopt_test_solvers solvers.cpp)
target_link_libraries(tinyopt_test_solvers PRIVATE ${THIRDPARTY_TEST_LIBS} tinyopt)
add_test_target(tinyopt_test_solvers)

add_executable(tinyopt_test_optimizers optimizers.cpp)
target_link_libraries(tinyopt_test_optimizers PRIVATE ${THIRDPARTY_TEST_LIBS} tinyopt)
add_test_target(tinyopt_test_optimizers)

add_executable(tinyopt_test_sparse sparse.cpp)
target_link_libraries(tinyopt_test_sparse PRIVATE ${THIRDPARTY_TEST_LIBS} tinyopt)
add_test_target(tinyopt_test_sparse)

add_executable(tinyopt_test_cov cov.cpp)
target_link_libraries(tinyopt_test_cov PRIVATE ${THIRDPARTY_TEST_LIBS} tinyopt)
add_test_target(tinyopt_test_cov)

add_executable(tinyopt_test_sqrt2 sqrt2.cpp)
target_link_libraries(tinyopt_test_sqrt2 PRIVATE ${THIRDPARTY_TEST_LIBS} tinyopt)
add_test_target(tinyopt_test_sqrt2)

add_executable(tinyopt_test_fit_circle circle.cpp)
target_link_libraries(tinyopt_test_fit_circle PRIVATE ${THIRDPARTY_TEST_LIBS} tinyopt)
add_test_target(tinyopt_test_fit_circle)

add_executable(tinyopt_test_userdef_params userdef_params.cpp)
target_link_libraries(tinyopt_test_userdef_params PRIVATE ${THIRDPARTY_TEST_LIBS} tinyopt)
add_test_target(tinyopt_test_userdef_params)

add_executable(tinyopt_test_userdef_params_jet userdef_params_jet.cpp)
target_link_libraries(tinyopt_test_userdef_params_jet PRIVATE ${THIRDPARTY_TEST_LIBS} tinyopt)
add_test_target(tinyopt_test_userdef_params_jet)

add_executable(tinyopt_test_nn nn.cpp)
target_link_libraries(tinyopt_test_nn PRIVATE ${THIRDPARTY_TEST_LIBS} tinyopt)
add_test_target(tinyopt_test_nn)

if(BUILD_TINYOPT_SOPHUS_EXAMPLES)
  add_executable(tinyopt_test_sophus sophus.cpp)
  target_link_libraries(tinyopt_test_sophus PRIVATE ${THIRDPARTY_TEST_LIBS} tinyopt)
  add_test_target(tinyopt_test_sophus)
endif(BUILD_TINYOPT_SOPHUS_EXAMPLES)

if(BUILD_TINYOPT_LIEPLUSPLUS_EXAMPLES)
  add_executable(tinyopt_test_lie++ lieplusplus.cpp)
  target_link_libraries(tinyopt_test_lie++ PRIVATE ${THIRDPARTY_TEST_LIBS} tinyopt)
  add_test_target(tinyopt_test_lie++)
endif(BUILD_TINYOPT_LIEPLUSPLUS_EXAMPLES)
