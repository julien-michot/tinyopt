#ifndef TINYOPT_API_H
#define TINYOPT_API_H

#include "conversions.h"
#include "generated_structs.h"
#include "structs.h"

#ifdef __cplusplus
extern "C" {
#endif

// Optimize a dyamically sized parameter structure ps
// using a user-defined residuals function (NLLS problem).
// 'opts' controls the optimization process.
// Numerical differentiation is used to compute gradients/hessians.
output_t optimize_res(params_t ps, res_func_t f, options_t opts);

// Optimize a dyamically sized parameter structure ps
// using a user-defined residuals function (NLLS problem).
// 'opts' controls the optimization process.
output_t optimize_res_grad(params_t ps, res_grad_func_t f, options_t opts);

// Optimize a dyamically sized parameter structure ps
// using a user-defined cost function (unconstrained problem).
// 'opts' controls the optimization process.
// Numerical differentiation is used to compute gradients/hessians.
output_t optimize_cost(params_t ps, cost_func_t f, options_t opts);

// Optimize a dyamically sized parameter structure ps
// using a user-defined cost function (unconstrained problem).
// 'opts' controls the optimization process.
output_t optimize_cost_grad(params_t ps, cost_grad_func_t f, options_t opts);

/* The C `_Generic` expression is invalid in C++; only provide this
   macro when building C consumers. */
#if !defined(__cplusplus)
#define optimize(ps, f, opts)             \
  _Generic((f),                           \
      res_func_t: optimize_res,           \
      res_grad_func_t: optimize_res_grad, \
      cost_func_t: optimize_cost,         \
      cost_grad_func_t: optimize_cost_grad)(ps, f, opts)
#endif

#ifdef __cplusplus
}
#endif

#endif