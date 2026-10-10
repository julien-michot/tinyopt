// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

#include <tinyopt/c/c_api_common.h>
#include <tinyopt/optimizers/options.h>

namespace tinyopt::c_api_detail {

static_assert(static_cast<int>(LinearSolverMethod::SuiteSparse) ==
              TINYOPT_LINEAR_SOLVER_SUITESPARSE);
static_assert(static_cast<int>(LinearSolverMethod::SVD) == TINYOPT_LINEAR_SOLVER_SVD);

inline tinyopt_options_t ToCOptions(const Options &source) {
  tinyopt_options_t result{};
  result.solver_type = static_cast<tinyopt_solver_t>(source.solver_type);
  result.linear_solver = static_cast<tinyopt_linear_solver_t>(source.linear_solver);
  result.svd_relative_threshold = source.svd_relative_threshold;
  result.save_history = source.save_history;
  result.measure_time = source.measure_time;
  result.use_step_quality_approx = source.opt.use_step_quality_approx;
  result.grad_clipping = source.opt.grad_clipping;
  result.hessian_is_full = source.hessian.H_is_full;
  result.check_min_hessian_diagonal = source.hessian.check_min_H_diag;
  result.save_last_hessian = source.hessian.save_last;
  result.use_squared_norm = source.cost.use_squared_norm;
  result.downscale_cost_by_two = source.cost.downscale_by_2;
  result.normalize_cost = source.cost.normalize;
  result.max_iters = source.stop.max_iters;
  result.min_error = source.stop.min_error;
  result.min_relative_error_decrease = source.stop.min_rerr_dec;
  result.min_step_norm_squared = source.stop.min_step_norm2;
  result.min_gradient_norm_squared = source.stop.min_grad_norm2;
  result.max_total_failures = source.stop.max_total_failures;
  result.max_consecutive_failures = source.stop.max_consec_failures;
  result.log_enabled = source.log.enable;
  result.log_error_symbol = nullptr;
  result.log_print_emoji = source.log.print_emoji;
  result.log_print_x = source.log.print_x;
  result.log_print_dx = source.log.print_dx;
  result.log_print_inliers = source.log.print_inliers;
  result.log_print_time = source.log.print_t;
  result.log_print_jacobian_jet = source.log.print_J_jet;
  result.log_print_max_standard_deviation = source.log.print_max_stdev;
  result.log_print_failure = source.log.print_failure;
  result.lm_jacobi_scaling = source.lm.jacobi_scaling;
  result.lm_damping_init = source.lm.damping_init;
  result.lm_damping_min = source.lm.damping_range[0];
  result.lm_damping_max = source.lm.damping_range[1];
  result.lm_good_factor = source.lm.good_factor;
  result.lm_bad_factor = source.lm.bad_factor;
  result.gd_learning_rate = source.gd.lr;
  result.cg_step_size = source.cg.step_size;
  result.cg_step_reduction = source.cg.step_reduction;
  result.dogleg_radius_init = source.dl.radius_init;
  result.dogleg_radius_max = source.dl.radius_max;
  result.dogleg_shrink_factor = source.dl.shrink_factor;
  result.dogleg_expand_factor = source.dl.expand_factor;
  result.bfgs_step_size = source.bfgs.step_size;
  result.bfgs_step_reduction = source.bfgs.step_reduction;
  result.bfgs_step_growth = source.bfgs.step_growth;
  result.bfgs_max_step_size = source.bfgs.max_step_size;
  result.bfgs_curvature_threshold = source.bfgs.curvature_threshold;
  result.lbfgs_step_size = source.lbfgs.step_size;
  result.lbfgs_step_reduction = source.lbfgs.step_reduction;
  result.lbfgs_step_growth = source.lbfgs.step_growth;
  result.lbfgs_max_step_size = source.lbfgs.max_step_size;
  result.lbfgs_curvature_threshold = source.lbfgs.curvature_threshold;
  result.lbfgs_history_size = source.lbfgs.history_size;
  return result;
}

inline Options ToTinyoptOptions(const tinyopt_options_t *source) {
  Options result;
  if (source == nullptr) return result;

  result.solver_type = static_cast<Options::Solver>(source->solver_type);
  result.linear_solver = static_cast<LinearSolverMethod>(source->linear_solver);
  result.svd_relative_threshold = source->svd_relative_threshold;
  result.save_history = source->save_history != 0;
  result.measure_time = source->measure_time != 0;
  result.opt.use_step_quality_approx = source->use_step_quality_approx != 0;
  result.opt.grad_clipping = source->grad_clipping;
  result.hessian.H_is_full = source->hessian_is_full != 0;
  result.hessian.check_min_H_diag = source->check_min_hessian_diagonal;
  result.hessian.save_last = source->save_last_hessian != 0;
  result.cost.use_squared_norm = source->use_squared_norm != 0;
  result.cost.downscale_by_2 = source->downscale_cost_by_two != 0;
  result.cost.normalize = source->normalize_cost != 0;
  result.stop.max_iters = source->max_iters;
  result.stop.min_error = source->min_error;
  result.stop.min_rerr_dec = source->min_relative_error_decrease;
  result.stop.min_step_norm2 = source->min_step_norm_squared;
  result.stop.min_grad_norm2 = source->min_gradient_norm_squared;
  result.stop.max_total_failures = source->max_total_failures;
  result.stop.max_consec_failures = source->max_consecutive_failures;
  result.stop.max_duration_ms = source->max_duration_ms;
  if (source->stop_callback != nullptr) {
    result.stop.stop_callback = [callback = source->stop_callback,
                                 user_data = source->stop_callback_user_data](
                                    double current, double previous, double gradient) {
      return callback(current, previous, gradient, user_data) != 0;
    };
  }
  if (source->stop_callback2 != nullptr) {
    result.stop.stop_callback2 = [callback = source->stop_callback2,
                                  user_data = source->stop_callback2_user_data](
                                     float error, const VecXf &x, const VecXf &dx) {
      return callback(error, x.data(), dx.data(), static_cast<int>(x.size()), user_data) != 0;
    };
  }
  if (source->step_callback != nullptr) {
    result.stop.step_callback = [callback = source->step_callback,
                                 user_data = source->step_callback_user_data](const VecXf &dx,
                                                                              bool is_rollback) {
      return callback(dx.data(), static_cast<int>(dx.size()), is_rollback ? 1 : 0, user_data) != 0;
    };
  }
  result.log.enable = source->log_enabled != 0;
  if (source->log_error_symbol != nullptr) result.log.e = source->log_error_symbol;
  result.log.print_emoji = source->log_print_emoji != 0;
  result.log.print_x = source->log_print_x != 0;
  result.log.print_dx = source->log_print_dx != 0;
  result.log.print_inliers = source->log_print_inliers != 0;
  result.log.print_t = source->log_print_time != 0;
  result.log.print_J_jet = source->log_print_jacobian_jet != 0;
  result.log.print_max_stdev = source->log_print_max_standard_deviation != 0;
  result.log.print_failure = source->log_print_failure != 0;
  result.lm.jacobi_scaling = source->lm_jacobi_scaling != 0;
  result.lm.damping_init = source->lm_damping_init;
  result.lm.damping_range = {source->lm_damping_min, source->lm_damping_max};
  result.lm.good_factor = source->lm_good_factor;
  result.lm.bad_factor = source->lm_bad_factor;
  result.gd.lr = source->gd_learning_rate;
  result.cg.step_size = source->cg_step_size;
  result.cg.step_reduction = source->cg_step_reduction;
  result.dl.radius_init = source->dogleg_radius_init;
  result.dl.radius_max = source->dogleg_radius_max;
  result.dl.shrink_factor = source->dogleg_shrink_factor;
  result.dl.expand_factor = source->dogleg_expand_factor;
  result.bfgs.step_size = source->bfgs_step_size;
  result.bfgs.step_reduction = source->bfgs_step_reduction;
  result.bfgs.step_growth = source->bfgs_step_growth;
  result.bfgs.max_step_size = source->bfgs_max_step_size;
  result.bfgs.curvature_threshold = source->bfgs_curvature_threshold;
  result.lbfgs.step_size = source->lbfgs_step_size;
  result.lbfgs.step_reduction = source->lbfgs_step_reduction;
  result.lbfgs.step_growth = source->lbfgs_step_growth;
  result.lbfgs.max_step_size = source->lbfgs_max_step_size;
  result.lbfgs.curvature_threshold = source->lbfgs_curvature_threshold;
  result.lbfgs.history_size = source->lbfgs_history_size;
  return result;
}

}  // namespace tinyopt::c_api_detail