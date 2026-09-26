/* Stable C binding header - hand-maintained
 * This file contains the minimal C-friendly types the C glue expects.
 * It is intentionally kept small and stable. The generator will write
 * a separate auto-generated header `generated_structs.h` in the build
 * tree for additional helper types; code should include this file for
 * the stable API.
 */
#ifndef TINYOPT_C_STRUCTS_H
#define TINYOPT_C_STRUCTS_H

#ifdef __cplusplus
extern "C" {
#endif

/* Basic C binding shapes expected by bindings/api.h */
typedef void (*plus_eq_t)(double* x, const double* dx);

typedef struct {
  double* x;  //< Pointer to the parameters
  int size;   //< Size of the parameters (must be > 0)

  plus_eq_t plus_eq;  //< Implement a Manifold delta x += dx
  int dims;           //< Dimension of the manifold, dx (or 0 is no manifold)
} params_t;
#define Ps (params_t)  // Macro to create params_t struct easily, e.g. Ps{{x, size}}

/* Function pointer types used by the C API */
typedef double (*cost_func_t)(const double* x);
typedef double (*cost_grad_func_t)(const double* x, double* grad_out);
typedef int (*res_cb_t)(const double* x, double* res_out);
typedef int (*res_grad_cb_t)(const double* x, double* res_out, double* grad_out,
                             double* hessian_out);

typedef struct {
  res_cb_t f;  // pointer to residuals function
  int nres;    // number of residuals
} res_func_t;

typedef struct {
  res_grad_cb_t f;  // pointer to residuals, gradient and hessian function
  int nres;         // number of residuals
} res_grad_func_t;

#ifdef __cplusplus
}
#endif

#endif /* TINYOPT_C_STRUCTS_H */
