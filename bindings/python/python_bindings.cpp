// nanobind-based Python bindings for a minimal tinyopt API
// This file exposes a convenience function `optimize_res_py`:
//   out_dict, x_final = optimize_res_py(x0: list[float], residuals_callable, nres: int, max_iters:
//   int=50)
// The Python residuals_callable is called as res = f(x_list) and must return an iterable of length
// nres.

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/vector.h>
#include <functional>
#include <memory>

// Avoid including the C API header here: it defines a C macro 'optimize' that
// can clash with C++ template/type resolution. Use the C++ API directly.
// Disable autodiff here to avoid template instantiations that expect Jet types
// when the Python-provided callable is not templated for autodiff.
#define TINYOPT_DISABLE_AUTODIFF
#include <tinyopt/diff/gradient_check.h>
#include <tinyopt/optimize.h>
#include <tinyopt/types.h>

#include "python_binding_helpers.h"

#include <tinyopt/bindings/params_wrapper.h>

namespace nb = nanobind;

// Generated struct bindings (from libclang code generator)
// If the generator produced a standalone translation unit, it should
// provide bind_generated_structs. In some build ordering situations the
// generated file may not be compiled separately; prefer including the
// generated source directly when available so the symbol is always
// present in this translation unit. Fall back to an inline no-op if the
// generated file is not present.
#if __has_include("generated_struct_bindings.cpp")
#include "generated_struct_bindings.cpp"
#endif

#ifndef TINYOPT_GENERATED_STRUCT_BINDINGS
// If the generated binding TU is not present or did not define the
// marker macro, provide a local no-op so the module still builds.
inline void bind_generated_structs(nb::module_ &m) { (void)m; }
#endif

using ParamsWrapper = tinyopt::bindings::ParamsWrapper;

// Cost/Residual evaluator adapters for Python callables. These mirror the
// embind/js-side adapters but use nanobind's call helper (`call_py_cast`) so
// we avoid unnecessary copies when passing Eigen types to/from Python.
struct CostEvaluator {
  explicit CostEvaluator(const nb::object &fn)
      : callable(fn), has_grad(py_callable_argcount(fn) > 1) {}

  double operator()(const ParamsWrapper &ps) const { return (*this)(ps.x); }

  double operator()(const tinyopt::VecX &x) const {
    if (has_grad)
      return call_py_cast<double>(callable, x, nb::none());
    else
      return call_py_cast<double>(callable, x);
  }

  template <typename G>
  double operator()(const ParamsWrapper &ps, G &g) const {
    if constexpr (traits::is_nullptr_v<G>) {
      return (*this)(ps);
    } else {
      if (has_grad) {  // manual gradient provided by Python callable
        Eigen::Ref<VecX> g_ref(g);
        return call_py_cast<double>(callable, ps.x, g_ref);
      } else {  // numeric gradient
        auto f = [this](const tinyopt::VecX &x) { return call_py_cast<double>(callable, x); };
        const auto &[res, J] = tinyopt::diff::NumEval(ps.x, f, tinyopt::diff::Method::kCentral);
        g = J.transpose() * res;
        return res;
      }
    }
  }

 private:
  nb::object callable = nb::none();
  const bool has_grad = false;
};

struct ResidualsEvaluator {
  explicit ResidualsEvaluator(const nb::object &fn)
      : callable(fn), has_grad_hessian(py_callable_argcount(fn) > 2) {}

  tinyopt::VecX operator()(const ParamsWrapper &ps) const { return (*this)(ps.x); }

  tinyopt::VecX operator()(const tinyopt::VecX &x) const {
    if (has_grad_hessian)
      return call_py_cast<tinyopt::VecX>(callable, x, nb::none(), nb::none());
    else
      return call_py_cast<tinyopt::VecX>(callable, x);
  }

  template <typename G, typename H>
  tinyopt::VecX operator()(const ParamsWrapper &ps, G &g, H &h) const {
    if constexpr (traits::is_nullptr_v<G>) {
      return (*this)(ps);
    } else {
      if (has_grad_hessian) {  // manual gradient/hessian provided by Python callable
        Eigen::Ref<MatX> g_ref(g);
        Eigen::Ref<MatX> h_ref(h);
        return call_py_cast<tinyopt::VecX>(callable, ps.x, g_ref, h_ref);
      } else {  // numeric gradient
        auto f = [this](const tinyopt::VecX &x) {
          return call_py_cast<tinyopt::VecX>(callable, x);
        };
        const auto &[res, J] = tinyopt::diff::NumEval(ps.x, f, tinyopt::diff::Method::kCentral);
        g = J.transpose() * res;
        h = J.transpose() * J;
        return res;
      }
    }
  }

 private:
  nb::object callable = nb::none();
  const bool has_grad_hessian = false;
};

// Helper: return the number of positional arguments declared by a Python
// callable, or -1 if it cannot be determined. This inspects the callable's
// __call__ (for callable objects) and reads __code__.co_argcount when
// available. The function acquires the GIL as it inspects Python objects.
// `py_callable_argcount` is implemented in python_binding_helpers.h

// helpers are provided in python_binding_helpers.h

NB_MODULE(tinyopt, m) {
  m.doc() = "Tinyopt minimal nanobind bindings (experimental)";

  bind_generated_structs(m);

  m.def(
      "optimize",
      [](const nb::ndarray<double> &x0, nb::object py_res_func,
         nb::object py_plus /* optional callable */, const tinyopt::Options &opts) {
        using namespace tinyopt;
        // Read x0 via the nanobind ndarray API
        if (x0.ndim() != 1) {
          throw std::runtime_error("x0 must be a 1-dimensional numpy array");
        }
        const size_t nx = x0.shape(0);
        const double *xptr = x0.data();

        ParamsWrapper x((double *)xptr, nx);
        if (!py_plus.is_none()) {
          // Keep a copy of the Python callable in the std::function capture.
          x.plus_manifold = [py_plus](const tinyopt::VecX &x_in, const tinyopt::VecX &delta) {
            return call_py_cast<tinyopt::VecX>(py_plus, x_in, delta);
          };
        }

        // Decide which callable to pass to tinyopt::Optimize. If the Python
        // residuals callable expects a single input argument (the params), we
        // pass the numerical-differentiation-backed `acc_func`. Otherwise we
        // assume the Python callable implements the full signature and pass
        // `residuals` directly.
        int res_num_params = py_callable_argcount(py_res_func);
        if (res_num_params <= 0) {
          TINYOPT_LOG("💡 couldn't determine callable arity ({}), assuming 1", res_num_params);
          res_num_params = 1;
        }
        const int is_nlls = opts.solver_type == tinyopt::Options::Solver::LevenbergMarquardt ||
                            opts.solver_type == tinyopt::Options::Solver::GaussNewton;

        // Call Optimize
        Output out;
        if (is_nlls) {
          auto f = ResidualsEvaluator(py_res_func);
          out = tinyopt::Optimize(x, f, opts);
        } else {
          auto f = CostEvaluator(py_res_func);
          out = tinyopt::Optimize(x, f, opts);
        }
        return nb::make_tuple(x.x, out);
      },
      nb::arg("x0"), nb::arg("residuals"), nb::arg("plus") = nb::none(),
      nb::arg("opts") = tinyopt::Options(),
      "Run optimization using the C++ API with a Python residuals callable (numpy arrays)");

  m.def(
      "check_gradient",
      [](const nb::ndarray<double> &x0, nb::object py_cost_func, nb::object py_plus) {
        // Read x0 via the nanobind ndarray API
        if (x0.ndim() != 1) {
          throw std::runtime_error("x0 must be a 1-dimensional numpy array");
        }

        ParamsWrapper x((double *)x0.data(), x0.shape(0));
        if (!py_plus.is_none()) {
          x.plus_manifold = [py_plus](const tinyopt::VecX &x_in, const tinyopt::VecX &delta) {
            return call_py_cast<tinyopt::VecX>(py_plus, x_in, delta);
          };
        }

        const int res_num_params = py_callable_argcount(py_cost_func);
        CostEvaluator acc_func(py_cost_func);
        return tinyopt::diff::CheckGradient(x, acc_func);
      },
      nb::arg("x0"), nb::arg("residuals"), nb::arg("plus") = nb::none(),
      "Run optimization using the C++ API with a Python residuals callable (numpy arrays)");
}
