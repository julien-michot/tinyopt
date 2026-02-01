
add_executable(tinyopt_test_traits traits.cpp)
add_test_target(tinyopt_test_traits)

add_executable(tinyopt_test_diff diff.cpp)
add_test_target(tinyopt_test_diff)

add_executable(tinyopt_test_basic basic.cpp)
add_test_target(tinyopt_test_basic)

add_executable(tinyopt_test_simple simple.cpp)
add_test_target(tinyopt_test_simple)

add_executable(tinyopt_test_types types.cpp)
add_test_target(tinyopt_test_types)

add_executable(tinyopt_test_solvers solvers.cpp)
add_test_target(tinyopt_test_solvers)

add_executable(tinyopt_test_optimizers optimizers.cpp)
add_test_target(tinyopt_test_optimizers)

add_executable(tinyopt_test_optimize_easy optimize_easy.cpp)
add_test_target(tinyopt_test_optimize_easy)

add_executable(tinyopt_test_optimize_hard optimize_hard.cpp)
add_test_target(tinyopt_test_optimize_hard)

add_executable(tinyopt_test_sparse sparse.cpp)
add_test_target(tinyopt_test_sparse)

add_executable(tinyopt_test_cov cov.cpp)
add_test_target(tinyopt_test_cov)

add_executable(tinyopt_test_sqrt2 sqrt2.cpp)
add_test_target(tinyopt_test_sqrt2)

add_executable(tinyopt_test_circle circle.cpp)
add_test_target(tinyopt_test_circle)

add_executable(tinyopt_test_userdef_params userdef_params.cpp)
add_test_target(tinyopt_test_userdef_params)

add_executable(tinyopt_test_userdef_params_jet userdef_params_jet.cpp)
add_test_target(tinyopt_test_userdef_params_jet)

add_executable(tinyopt_test_nn nn.cpp)
add_test_target(tinyopt_test_nn)

add_executable(tinyopt_test_check_gradient check_gradient.cpp)
add_test_target(tinyopt_test_check_gradient)

# Unconstrained Optimization

add_executable(tinyopt_test_unconstrained unconstrained.cpp)
add_test_target(tinyopt_test_unconstrained)


# Losses

add_executable(tinyopt_test_norms norms.cpp)
add_test_target(tinyopt_test_norms)

add_executable(tinyopt_test_robust_norms robust_norms.cpp)
add_test_target(tinyopt_test_robust_norms)

add_executable(tinyopt_test_activations activations.cpp)
add_test_target(tinyopt_test_activations)

add_executable(tinyopt_test_classif classif.cpp)
add_test_target(tinyopt_test_classif)

add_executable(tinyopt_test_mahalanobis mahalanobis.cpp)
add_test_target(tinyopt_test_mahalanobis)

add_executable(tinyopt_test_distances distances.cpp)
add_test_target(tinyopt_test_distances)

# 3rd parties

if(TINYOPT_BUILD_SOPHUS_TEST)
  add_executable(tinyopt_test_sophus sophus.cpp)
  add_test_target(tinyopt_test_sophus)
endif(TINYOPT_BUILD_SOPHUS_TEST)

if(TINYOPT_BUILD_LIEPLUSPLUS_TEST)
  add_executable(tinyopt_test_lie++ lieplusplus.cpp)
  add_test_target(tinyopt_test_lie++)
endif(TINYOPT_BUILD_LIEPLUSPLUS_TEST)

if (TINYOPT_BUILD_CERES)
  add_executable(tinyopt_test_ceres ceres.cpp)
  add_test_target(tinyopt_test_ceres)
endif()
