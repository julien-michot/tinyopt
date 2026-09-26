#pragma once

#include <nanobind/nanobind.h>
#include <iostream>
#include <utility>
#include <vector>

#include <tinyopt/diff/gradient_check.h>
#include <tinyopt/optimize.h>
#include <tinyopt/types.h>

namespace nb = nanobind;

// ParamsWrapper is shared across bindings.
#include <tinyopt/bindings/params_wrapper.h>

using ParamsWrapper = tinyopt::bindings::ParamsWrapper;

// (Num-diff-backed accumulators were removed — the Python bindings now use
// lightweight evaluator adapters defined in the translation unit to avoid
// duplicating the NumEval wiring.)

// Helper to call a Python callable and cast the result safely while holding the GIL.
template <typename Ret, typename... Args>
static Ret call_py_cast(const nb::object &f, Args &&...args) {
  try {
    nb::gil_scoped_acquire acquire;
    return nb::cast<Ret>(f(std::forward<Args>(args)...));
  } catch (const std::exception &e) {
    std::cerr << "residuals wrapper exception: " << e.what() << std::endl;
    throw;
  }
}

// Manual-callable adapters for the different callable signatures we support.
// NOTE: manual-callable adapters that access ParamsWrapper::x are defined
// in the .cpp where ParamsWrapper is a complete type.

// Helper: return the number of positional arguments declared by a Python
// callable, or -1 if it cannot be determined. This inspects the callable's
// __call__ (for callable objects) and reads __code__.co_argcount when
// available. The function acquires the GIL as it inspects Python objects.
static int py_callable_argcount(const nb::object &py_callable) {
  nb::gil_scoped_acquire acquire;
  try {
    nb::object target = py_callable;

    try {
      if (nb::hasattr(target, "func")) {
        target = target.attr("func");
      }
    } catch (...) {
    }

    try {
      while (nb::hasattr(target, "__wrapped__")) {
        try {
          target = target.attr("__wrapped__");
        } catch (...) {
          break;
        }
      }
    } catch (...) {
    }

    try {
      if (nb::hasattr(target, "__code__")) {
        auto code = target.attr("__code__");
        return nb::cast<int>(code.attr("co_argcount"));
      }
    } catch (...) {
    }

    try {
      nb::module_ inspect = nb::module_::import_("inspect");
      auto sig = inspect.attr("signature")(target);
      auto params = sig.attr("parameters");
      auto vals = nb::cast<std::vector<nb::object>>(params.attr("values")());
      const int POS_ONLY = nb::cast<int>(inspect.attr("_POSITIONAL_ONLY"));
      const int POS_OR_KEY = nb::cast<int>(inspect.attr("_POSITIONAL_OR_KEYWORD"));
      int count = 0;
      for (auto &p : vals) {
        try {
          int kind = nb::cast<int>(p.attr("kind"));
          if (kind == POS_ONLY || kind == POS_OR_KEY) ++count;
        } catch (...) {
        }
      }
      if (count > 0) return count;
    } catch (...) {
    }

    try {
      if (nb::hasattr(target, "__call__")) {
        try {
          auto call_attr = target.attr("__call__");
          if (nb::hasattr(call_attr, "__code__")) {
            auto code = call_attr.attr("__code__");
            return nb::cast<int>(code.attr("co_argcount"));
          }
          try {
            nb::module_ inspect = nb::module_::import_("inspect");
            auto sig = inspect.attr("signature")(call_attr);
            auto params = sig.attr("parameters");
            auto vals = nb::cast<std::vector<nb::object>>(params.attr("values")());
            const int POS_ONLY = nb::cast<int>(inspect.attr("_POSITIONAL_ONLY"));
            const int POS_OR_KEY = nb::cast<int>(inspect.attr("_POSITIONAL_OR_KEYWORD"));
            int count = 0;
            for (auto &p : vals) {
              try {
                int kind = nb::cast<int>(p.attr("kind"));
                if (kind == POS_ONLY || kind == POS_OR_KEY) ++count;
              } catch (...) {
              }
            }
            if (count > 0) return count;
          } catch (...) {
          }
        } catch (...) {
        }
      }
    } catch (...) {
    }
  } catch (...) {
  }
  return -1;
}
