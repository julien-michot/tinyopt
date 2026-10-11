
add_executable(tinyopt_test_traits traits.cpp)
add_test_target(tinyopt_test_traits)

add_executable(tinyopt_test_logging logging.cpp)
add_test_target(tinyopt_test_logging)

add_executable(tinyopt_test_diff diff.cpp)
add_test_target(tinyopt_test_diff)

add_executable(tinyopt_test_basic basic.cpp)
add_test_target(tinyopt_test_basic)

add_executable(tinyopt_test_simple simple.cpp
                                   explicit_instantiation_impl.cpp)
add_test_target(tinyopt_test_simple)

add_executable(tinyopt_test_types types.cpp)
add_test_target(tinyopt_test_types)

add_executable(tinyopt_test_solver_lm solver_lm.cpp)
add_test_target(tinyopt_test_solver_lm)

add_executable(tinyopt_test_solver_gn solver_gn.cpp)
add_test_target(tinyopt_test_solver_gn)

add_executable(tinyopt_test_solver_gd solver_gd.cpp)
add_test_target(tinyopt_test_solver_gd)

add_executable(tinyopt_test_solver_cg solver_cg.cpp)
add_test_target(tinyopt_test_solver_cg)

add_executable(tinyopt_test_solver_dl solver_dl.cpp)
add_test_target(tinyopt_test_solver_dl)

add_executable(tinyopt_test_solver_bfgs solver_bfgs.cpp)
add_test_target(tinyopt_test_solver_bfgs)

add_executable(tinyopt_test_linear_solvers linear_solvers.cpp)
add_test_target(tinyopt_test_linear_solvers)

add_executable(tinyopt_test_optimizer_lm optimizer_lm.cpp)
add_test_target(tinyopt_test_optimizer_lm)

add_executable(tinyopt_test_optimizer_gn optimizer_gn.cpp)
add_test_target(tinyopt_test_optimizer_gn)

add_executable(tinyopt_test_optimizer_gd optimizer_gd.cpp)
add_test_target(tinyopt_test_optimizer_gd)

add_executable(tinyopt_test_optimizer_cg optimizer_cg.cpp)
add_test_target(tinyopt_test_optimizer_cg)

add_executable(tinyopt_test_optimizer_dl optimizer_dl.cpp)
add_test_target(tinyopt_test_optimizer_dl)

add_executable(tinyopt_test_optimizer_bfgs optimizer_bfgs.cpp)
add_test_target(tinyopt_test_optimizer_bfgs)

add_executable(tinyopt_test_optimize_easy optimize_easy.cpp)
add_test_target(tinyopt_test_optimize_easy)

add_executable(tinyopt_test_variadic variadic.cpp)
add_test_target(tinyopt_test_variadic)

add_executable(tinyopt_test_dynamic_alloc dynamic_alloc.cpp)
add_test_target(tinyopt_test_dynamic_alloc)

add_executable(tinyopt_test_parameter_pack parameter_pack.cpp)
add_test_target(tinyopt_test_parameter_pack)

add_executable(tinyopt_test_tutorial tutorial.cpp)
add_test_target(tinyopt_test_tutorial)

add_executable(tinyopt_test_params_wrapper params_wrapper.cpp)
add_test_target(tinyopt_test_params_wrapper)

add_executable(tinyopt_test_optimize_hard optimize_hard.cpp)
add_test_target(tinyopt_test_optimize_hard)

add_executable(tinyopt_test_sparse sparse.cpp)
add_test_target(tinyopt_test_sparse)

add_executable(tinyopt_test_cov cov.cpp)
add_test_target(tinyopt_test_cov)

if (NOT MSVC) # msvc keep segfaulting on this test, even with /permissive- and /std:c++latest
  add_executable(tinyopt_test_sqrt2 sqrt2.cpp)
  add_test_target(tinyopt_test_sqrt2)
endif()

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

add_executable(tinyopt_test_locked locked.cpp)
add_test_target(tinyopt_test_locked)

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

if (TINYOPT_ENABLE_SUITESPARSE)
  add_executable(tinyopt_test_suitesparse suitesparse.cpp)
  add_test_target(tinyopt_test_suitesparse)
endif()
