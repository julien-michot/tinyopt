

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

if(TINYOPT_BUILD_SOPHUS_EXAMPLES)
  add_executable(tinyopt_test_sophus sophus.cpp)
  target_link_libraries(tinyopt_test_sophus PRIVATE ${THIRDPARTY_TEST_LIBS} tinyopt)
  add_test_target(tinyopt_test_sophus)
endif(TINYOPT_BUILD_SOPHUS_EXAMPLES)

if(TINYOPT_BUILD_LIEPLUSPLUS_EXAMPLES)
  add_executable(tinyopt_test_lie++ lieplusplus.cpp)
  target_link_libraries(tinyopt_test_lie++ PRIVATE ${THIRDPARTY_TEST_LIBS} tinyopt)
  add_test_target(tinyopt_test_lie++)
endif(TINYOPT_BUILD_LIEPLUSPLUS_EXAMPLES)

# Losses

add_executable(tinyopt_test_norms norms.cpp)
target_link_libraries(tinyopt_test_norms PRIVATE ${THIRDPARTY_TEST_LIBS} tinyopt)
add_test_target(tinyopt_test_norms)

add_executable(tinyopt_test_robust_norms robust_norms.cpp)
target_link_libraries(tinyopt_test_robust_norms PRIVATE ${THIRDPARTY_TEST_LIBS} tinyopt)
add_test_target(tinyopt_test_robust_norms)

add_executable(tinyopt_test_activations activations.cpp)
target_link_libraries(tinyopt_test_activations PRIVATE ${THIRDPARTY_TEST_LIBS} tinyopt)
add_test_target(tinyopt_test_activations)

add_executable(tinyopt_test_classif classif.cpp)
target_link_libraries(tinyopt_test_classif PRIVATE ${THIRDPARTY_TEST_LIBS} tinyopt)
add_test_target(tinyopt_test_classif)

add_executable(tinyopt_test_mahalanobis mahalanobis.cpp)
target_link_libraries(tinyopt_test_mahalanobis PRIVATE ${THIRDPARTY_TEST_LIBS} tinyopt)
add_test_target(tinyopt_test_mahalanobis)

add_executable(tinyopt_test_distances distances.cpp)
target_link_libraries(tinyopt_test_distances PRIVATE ${THIRDPARTY_TEST_LIBS} tinyopt)
add_test_target(tinyopt_test_distances)

# 3rd parties
if (TINYOPT_BUILD_CERES)
  add_executable(tinyopt_test_ceres ceres.cpp)
  target_link_libraries(tinyopt_test_ceres PRIVATE ${THIRDPARTY_TEST_LIBS} tinyopt)
  add_test_target(tinyopt_test_ceres)
endif()
