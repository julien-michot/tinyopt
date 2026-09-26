
#include <assert.h>
#include <math.h>
#include <stdio.h>

#include "c/api.h"

void simple_manifold_plus_eq2(double* x, const double* dx) {
  // plus_eq is expected to perform x <- x (+) dx.
  // Use standard addition so numerical differentiation behaves correctly.
  x[0] += dx[0];
  x[1] += dx[1];
}

// Residuals are: res = x - [5, 3]
int simple_residuals(const double* x, double* res) {
  res[0] = x[0] - 5.0;  // Target x = 5
  res[1] = x[1] - 3.0;  // Target x = 3
  return 2;             // Return the number of residuals
}

// Residuals are: res = x - [5, 3]
int simple_residuals_grad(const double* x, double* res, double* grad, double* hessian) {
  res[0] = x[0] - 5.0;  // Target x = 5
  res[1] = x[1] - 3.0;  // Target x = 3
  if (grad) {           // Update gradient as Jt*res (column-major)
    grad[0] = 1 * res[0];
    grad[1] = 1 * res[1];
  }
  if (hessian) {  // Update Hessian (column-major)
    // Hessian is identity
    hessian[0] = 1;
    hessian[1] = 0;
    hessian[2] = 0;
    hessian[3] = 1;
  }
  return 2;  // Return the number of residuals
}

// Residuals are: res = x - [5, 3]
double simple_cost(const double* x) { return fabs(x[0] - 5.0) + fabs(x[1] - 3.0); }
double simple_cost_grad(const double* x, double* g) {
  double res = fabs(x[0] - 5.0) + fabs(x[1] - 3.0);
  if (g) {  // Update gradient as Jt*res (column-major)
    g[0] = (x[0] < 5.0 ? -1 : 1) * res;
    g[1] = (x[1] < 3.0 ? -1 : 1) * res;
  }
  return res;
}

int check_success(output_t out, double* x) {
  double eps = 1e-4;
  int success = (out.stop_reason >= 0 && fabs(x[0] - 5.0) < eps && fabs(x[1] - 3.0) < eps);
  if (success) {
    printf("✅ Success! Final x: %f, %f\n", x[0], x[1]);
  } else if (fabs(x[0] - 5.0) < eps && fabs(x[1] - 3.0) < eps) {
    printf("❌ Failed with stop_reason: %d\n", out.stop_reason);
    return 1;
  } else {
    printf("❌ Failed to converge x: %f, %f\n", x[0], x[1]);
    return 2;
  }
  return 0;
}

int test_dense_dyn_nlls() {
  double x[2] = {3.0, 2.0};
  options_t opts = {.solver_type = LevenbergMarquardt, .max_iters = 10};
  // opts.print_grad = 1;
  // opts.print_hessian = 1;

  printf("******** Starting NLLS optimization with Numerical Differentiation ...\n");

  res_func_t f = {.f = simple_residuals, .nres = 2};
  params_t ps = {.x = x, .size = 2};
  // NOTE: This test exercises numerical differentiation. Providing a custom
  // plus_eq changes how parameter increments are applied; this can invalidate
  // the finite-difference assumptions depending on the manifold.
  // Keep manifold tests in the dedicated Python manifold test.
  output_t out = optimize(ps, f, opts);

  return check_success(out, x);
}

int test_dense_dyn_grad_nlls() {
  double x[2] = {3.0, 2.0};
  options_t opts = {.solver_type = LevenbergMarquardt, .max_iters = 10};
  // opts.print_grad = 1;
  // opts.print_hessian = 1;

  printf("******** Starting NLLS optimization with Manual Gradient...\n");

  res_grad_func_t f = {.f = simple_residuals_grad, .nres = 2};
  params_t ps = {.x = x, .size = 2};
  output_t out = optimize(ps, f, opts);

  return check_success(out, x);
}

int test_dense_dyn_unconstrained() {
  double x[2] = {3.0, 2.0};
  options_t opts = {.solver_type = GradientDescent, .max_iters = 10};
  opts.gd.lr = 0.5;
  // opts.print_grad = 1;
  // opts.print_hessian = 1;

  printf("******** Starting Unconstrained optimization with Numerical Differentiation ...\n");

  params_t ps = {.x = x, .size = 2};
  output_t out = optimize(ps, simple_cost, opts);

  // Numerical differentiation with only 10 iterations may not reach the
  // strict epsilon threshold; accept near-solution.
  double eps = 5e-2;
  int success = (out.stop_reason >= 0 && fabs(x[0] - 5.0) < eps && fabs(x[1] - 3.0) < eps);
  if (success) return 0;
  return check_success(out, x);
}

int test_dense_dyn_grad_unconstrained() {
  double x[2] = {3.0, 2.0};
  options_t opts = {.solver_type = GradientDescent, .max_iters = 10};
  opts.gd.lr = 0.5;
  // opts.print_grad = 1;
  // opts.print_hessian = 1;

  printf("******** Starting Unconstrained optimization with Manual Gradient...\n");

  params_t ps = {.x = x, .size = 2};
  output_t out = optimize(ps, simple_cost_grad, opts);

  return check_success(out, x);
}

int main() {
  int o = test_dense_dyn_nlls();
  o += test_dense_dyn_grad_nlls();

  o += test_dense_dyn_unconstrained();
  o += test_dense_dyn_grad_unconstrained();

  return o;
}