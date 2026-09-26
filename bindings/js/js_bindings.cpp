// Emscripten/Embind-based JavaScript bindings for tinyopt
// This file exposes a convenience function `optimize`:
//   result = tinyopt.optimize(x0, residuals_callable, opts)
// The JavaScript residuals_callable is called as res = f(x) and must return an array.

#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <cmath>
#include <functional>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

// Disable autodiff to avoid template instantiations that expect Jet types
// when the JavaScript-provided callable is not templated for autodiff.
#define TINYOPT_DISABLE_AUTODIFF
#include <tinyopt/diff/gradient_check.h>
#include <tinyopt/optimize.h>
#include <tinyopt/traits.h>
#include <tinyopt/types.h>

#include <tinyopt/bindings/params_wrapper.h>

// Include generated Embind bindings and conversion functions
#include "generated_embind_bindings.cpp"

namespace em = emscripten;
using namespace tinyopt;

using ParamsWrapper = tinyopt::bindings::ParamsWrapper;

// Helper: convert JavaScript array/typed array to std::vector<double>
std::vector<double> js_to_vector(const em::val &js_array) {
  std::vector<double> result;
  unsigned int length = js_array["length"].as<unsigned int>();
  result.reserve(length);
  for (unsigned int i = 0; i < length; ++i) {
    result.push_back(js_array[i].as<double>());
  }
  return result;
}

// Helper: convert Eigen vector to JavaScript typed array (zero-copy when possible)
em::val eigen_to_js_array(const tinyopt::VecX &vec) {
  // Create a typed array view directly from Eigen's memory
  // Note: The data must remain valid while the JS array is in use
  return em::val(em::typed_memory_view(vec.size(), vec.data()));
}

// Helper: convert Eigen matrix to JavaScript typed array or array-of-arrays.
// If `copy` is false (default) this returns a flat TypedArray view (Float64Array)
// over the matrix storage (rows*cols, column-major order for Eigen). The
// returned view points to the C++ memory — the memory must remain valid while
// JS uses the view. If `copy` is true, a new JS array-of-arrays is created and
// populated with row-major access (each sub-array is a JS Array of numbers).
em::val eigen_to_js_array(const tinyopt::MatX &mat, bool copy = false) {
  if (!copy) {
    // Return a flat typed array view (zero-copy). Note: Eigen uses column-major
    // storage by default; the JS consumer will receive the raw memory as a
    // flat Float64Array of length rows*cols.
    return em::val(em::typed_memory_view(mat.size(), mat.data()));
  }

  // Copy mode: create an array of rows, each row is a JS array of numbers.
  em::val js_rows = em::val::array();
  const int rows = static_cast<int>(mat.rows());
  const int cols = static_cast<int>(mat.cols());
  for (int r = 0; r < rows; ++r) {
    std::vector<double> row_vec;
    row_vec.reserve(cols);
    for (int c = 0; c < cols; ++c) {
      row_vec.push_back(mat(r, c));
    }
    js_rows.set(r, em::val::array(row_vec.begin(), row_vec.end()));
  }
  return js_rows;
}

// Detect the number of parameters a JavaScript function accepts
int js_callable_argcount(const em::val &callable) {
  if (callable.isNull() || callable.isUndefined()) return -1;
  // Check if it's a function
  if (!callable.instanceof (em::val::global("Function"))) return -1;
  // Try to get the length property (number of declared parameters)
  if (callable.hasOwnProperty("length")) {
    return callable["length"].as<int>();
  }
  return -1;
}

struct CostEvaluator {
  explicit CostEvaluator(const em::val &fn)
      : callable(fn), has_grad(js_callable_argcount(fn) > 1) {}
  double operator()(const ParamsWrapper &ps) const { return (*this)(eigen_to_js_array(ps.x)); }

  double operator()(const em::val &js_x) const {
    if (has_grad)
      return callable(js_x, em::val::null()).as<double>();
    else
      return callable(js_x).as<double>();
  }

  template <typename G>
  double operator()(const ParamsWrapper &ps, G &g) const {
    if constexpr (tinyopt::traits::is_nullptr_v<G>) {
      return (*this)(ps);
    } else {
      const auto js_x = eigen_to_js_array(ps.x);
      if (has_grad) {  // manual gradient
        auto js_g = eigen_to_js_array(g);
        return callable(js_x, js_g).template as<double>();
      } else {  // numeric gradient
        auto f = [this](const tinyopt::VecX &x) {
          const auto js_x = eigen_to_js_array(x);
          return callable(js_x).as<double>();
        };
        const auto &[res, J] = tinyopt::diff::NumEval(ps.x, f, tinyopt::diff::Method::kCentral);
        g = J.transpose() * res;
        return res;
      }
    }
  }

 private:
  em::val callable = em::val::undefined();
  const bool has_grad = false;
};

struct ResidualsEvaluator {
  explicit ResidualsEvaluator(const em::val &fn)
      : callable(fn), has_grad_hessian(js_callable_argcount(fn) > 2) {}

  static tinyopt::VecX convert(const em::val &js_res) {
    // Convert JS array/typed array to std::vector then to Eigen vector (copy)
    const auto vec = js_to_vector(js_res);
    return tinyopt::VecX(Eigen::Map<const tinyopt::VecX>(vec.data(), vec.size()));
  }

  tinyopt::VecX operator()(const ParamsWrapper &ps) const {
    return (*this)(eigen_to_js_array(ps.x));
  }

  tinyopt::VecX operator()(const em::val &js_x) const {
    if (has_grad_hessian)
      return convert(callable(js_x, em::val::null(), em::val::null()));
    else
      return convert(callable(js_x));
  }

  template <typename G, typename H>
  tinyopt::VecX operator()(const ParamsWrapper &ps, G &g, H &h) const {
    if constexpr (tinyopt::traits::is_nullptr_v<G>) {
      return (*this)(ps);
    } else {
      const auto js_x = eigen_to_js_array(ps.x);
      if (has_grad_hessian) {  // manual gradient
        auto js_g = eigen_to_js_array(g);
        auto js_h = eigen_to_js_array(h);
        return convert(callable(js_x, js_g, js_h));
      } else {  // numeric gradient
        auto f = [this](const tinyopt::VecX &x) {
          const auto js_x = eigen_to_js_array(x);
          return convert(callable(js_x));
        };
        const auto &[res, J] = tinyopt::diff::NumEval(ps.x, f, tinyopt::diff::Method::kCentral);
        g = J.transpose() * res;
        h = J.transpose() * J;
        return res;
      }
    }
  }

 private:
  em::val callable = em::val::undefined();
  const bool has_grad_hessian = false;
};

// Main optimize function exposed to JavaScript
em::val optimize_js(const em::val &x0_js, const em::val &residuals_js, const em::val &plus_js,
                    const tinyopt::Options &opts) {
  using namespace tinyopt;

  // Convert JavaScript array to std::vector
  auto x0_vec = js_to_vector(x0_js);

  ParamsWrapper x(x0_vec.data(), x0_vec.size());
  if (!plus_js.isNull() && !plus_js.isUndefined()) {
    x.plus_manifold = [plus_js](const tinyopt::VecX &x_in, const tinyopt::VecX &delta) {
      auto result = plus_js(eigen_to_js_array(x_in), eigen_to_js_array(delta));
      auto result_vec = em::vecFromJSArray<double>(result);
      return tinyopt::VecX(Eigen::Map<const tinyopt::VecX>(result_vec.data(), result_vec.size()));
    };
  }

  const bool is_nlls = opts.solver_type == tinyopt::Options::Solver::LevenbergMarquardt ||
                       opts.solver_type == tinyopt::Options::Solver::GaussNewton;

  // Call Optimize with runtime exception logging so JS sees helpful error messages
  Output out;
  // Log detection info to JS console for debugging
  try {
    std::string dbg =
        "[optimize_js] fn_num_args=" + std::to_string(js_callable_argcount(residuals_js)) +
        ", solver_type=" + std::to_string(static_cast<int>(opts.solver_type)) +
        ", is_nlls=" + std::to_string(is_nlls);
    em::val::global("console").call<void>("log", dbg);
  } catch (...) {
  }
  try {
    if (is_nlls) {
      auto f = ResidualsEvaluator(residuals_js);
      out = tinyopt::Optimize(x, f, opts);
    } else {
      auto f = CostEvaluator(residuals_js);
      out = tinyopt::Optimize(x, f, opts);
    }
  } catch (const std::exception &e) {
    try {
      em::val::global("console").call<void>("error",
                                            std::string("C++ exception in Optimize: ") + e.what());
    } catch (...) {
    }
    throw;
  } catch (...) {
    try {
      em::val::global("console").call<void>("error",
                                            std::string("Unknown C++ exception in Optimize"));
    } catch (...) {
    }
    throw;
  }

  // Return result as JavaScript object with x_final and output
  em::val result = em::val::object();
  // Copy the final x to a JavaScript array (cannot use typed_memory_view for owned data)
  result.set("x", eigen_to_js_array(x.x, true));

  // Use generated conversion function for Output
  em::val out_obj = convert_output_to_js(out);

  // Add per-iteration results if available (not in generated conversion yet)
  if (!out.errs.empty()) {
    out_obj.set("errors", em::val::array(out.errs.begin(), out.errs.end()));
  }

  result.set("output", out_obj);

  return result;
}

// Gradient check function
bool check_gradient_js(const em::val &x0_js, const em::val &cost_func_js, const em::val &plus_js) {
  auto x0_vec = js_to_vector(x0_js);
  ParamsWrapper x(x0_vec.data(), x0_vec.size());

  if (!plus_js.isNull() && !plus_js.isUndefined()) {
    x.plus_manifold = [plus_js](const tinyopt::VecX &x_in, const tinyopt::VecX &delta) {
      auto result = plus_js(eigen_to_js_array(x_in), eigen_to_js_array(delta));
      auto result_vec = em::vecFromJSArray<double>(result);
      return tinyopt::VecX(Eigen::Map<const tinyopt::VecX>(result_vec.data(), result_vec.size()));
    };
  }

  auto f = CostEvaluator(cost_func_js);
  return tinyopt::diff::CheckGradient(x, f);
}

// Embind bindings
EMSCRIPTEN_BINDINGS(tinyopt_module) {
  using namespace tinyopt;

  // Bind generated structs (Options, Output, enums, etc.)
  bind_generated_embind_structs();

  // Bind generated conversion functions
  bind_generated_embind_conversions();

  // Bind main functions
  em::function("optimize", &optimize_js, em::allow_raw_pointers());

  em::function("check_gradient", &check_gradient_js, em::allow_raw_pointers());
}
