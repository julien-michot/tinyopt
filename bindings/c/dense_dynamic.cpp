#include "api.h"

#include <tinyopt/optimize.h>

#include <Eigen/Core>

#include <cstring>
#include <vector>

// Adapter for the C API `params_t`.
//
// Key property: copy-less view of the user-provided parameter buffer.
// The lifetime of `ps` is managed by the caller; this wrapper only borrows it.
struct ParamsWrapper {
  using Scalar = double;

  explicit ParamsWrapper(const params_t &ps) : ps_(ps), x_map_(ps_.x, ps_.size) {}

  // NumDiff relies on copying the params object (X_t y = x) and perturbing y.
  // For the C API, `ps_.x` points to caller-owned memory; a default copy would
  // alias the same buffer and break numerical differentiation.
  //
  // Make the wrapper copyable by deep-copying the backing buffer when we own it.
  ParamsWrapper(const ParamsWrapper &other)
      : ps_(other.ps_),
        owned_storage_(static_cast<size_t>(other.ps_.size)),
        x_map_(owned_storage_.data(), other.ps_.size) {
    std::memcpy(owned_storage_.data(), other.ps_.x,
                static_cast<size_t>(other.ps_.size) * sizeof(double));
    ps_.x = owned_storage_.data();
  }

  ParamsWrapper &operator=(const ParamsWrapper &other) {
    if (this == &other) return *this;
    ps_ = other.ps_;
    owned_storage_.resize(static_cast<size_t>(other.ps_.size));
    std::memcpy(owned_storage_.data(), other.ps_.x,
                static_cast<size_t>(other.ps_.size) * sizeof(double));
    ps_.x = owned_storage_.data();
    x_map_ = Eigen::Map<tinyopt::VecX>(ps_.x, ps_.size);
    return *this;
  }

  int size() const { return ps_.size; }
  int dims() const { return ps_.dims > 0 ? ps_.dims : ps_.size; }

  double *data() { return ps_.x; }
  const double *data() const { return ps_.x; }

  ParamsWrapper &operator+=(const tinyopt::VecX &delta) {
    if (ps_.plus_eq) {
      ps_.plus_eq(ps_.x, delta.data());
    } else {
      x_map_ += delta;
    }
    return *this;
  }

  bool is_valid() const {
    if (!ps_.x || ps_.size <= 0) {
      TINYOPT_LOG("❌ Invalid sizes: ps.x = {}, ps.size = {}", (void *)ps_.x, ps_.size);
      return false;
    }
    if (ps_.plus_eq && (ps_.dims == 0 || ps_.dims > ps_.size)) {
      TINYOPT_LOG("❌ Invalid manifold dims: {}", ps_.dims);
      return false;
    }
    return true;
  }

  params_t ps_;
  // When non-empty, this wrapper owns `ps_.x` and points it to this buffer.
  // Construction from `params_t` borrows caller memory (empty storage).
  // Copy construction deep-copies into owned storage.
  std::vector<double> owned_storage_;
  Eigen::Map<tinyopt::VecX> x_map_;
};

extern "C" {

output_t optimize_res(params_t ps, res_func_t cb, options_t opts) {
  ParamsWrapper x(ps);
  if (!x.is_valid()) return output_t{.stop_reason = -100};
  if (!cb.f || cb.nres <= 0) {
    TINYOPT_LOG("❌ Cost function missing cb = {} nres: {}", (void *)cb.f, cb.nres);
    return output_t{.stop_reason = -100};  // Invalid input
  }

  // Wrap C callback into a C++ lambda for the template
  auto cpp_cb = [cb](auto &x) {
    tinyopt::VecX res(cb.nres);
    int n = cb.f(x.data(), res.data());
    assert(n >= 0 && "Callback failed");
    return res;
  };
  // Use numerical differentiation to build the accurate function
  auto acc_func = [cpp_cb, &opts](auto &x, auto &g, auto &h) {
    if constexpr (!tinyopt::traits::is_nullptr_v<decltype(g)>) {
      const auto &[res, J] = tinyopt::diff::NumEval(x, cpp_cb, tinyopt::diff::Method::kCentral);
      g = J.transpose() * res;
      h = J.transpose() * J;
      // if (opts.print_grad) TINYOPT_LOG("💡 grad: {}", g);
      // if (opts.print_hessian) TINYOPT_LOG("💡 H: {}", h);
      return res;
    } else {
      return cpp_cb(x);
    }
  };

  // Call the C++ template
  auto result = tinyopt::Optimize(x, acc_func, Convert(opts));
  return Convert(result);
}

output_t optimize_res_grad(params_t ps, res_grad_func_t cb, options_t opts) {
  ParamsWrapper x(ps);
  if (!x.is_valid()) return output_t{.stop_reason = -100};

  if (!cb.f || cb.nres <= 0) {
    TINYOPT_LOG("❌ Cost function missing cb = {} nres: {}", (void *)cb.f, cb.nres);
    return output_t{.stop_reason = -100};  // Invalid input
  }

  // Wrap C callback into a C++ lambda for the template
  auto acc_func = [cb, &opts](auto &x, auto &g, auto &h) {
    tinyopt::VecX res(cb.nres);
    int n;
    if constexpr (!tinyopt::traits::is_nullptr_v<decltype(g)>) {
      n = cb.f(x.data(), res.data(), g.data(), h.data());
      // if (opts.print_grad) TINYOPT_LOG("💡 grad: {}", g);
      // if (opts.print_hessian) TINYOPT_LOG("💡 H: {}", h);

    } else {
      n = cb.f(x.data(), res.data(), nullptr, nullptr);
    }
    assert(n >= 0 && "Callback failed");
    return res;
  };

  // Call the C++ template
  auto result = tinyopt::Optimize(x, acc_func, Convert(opts));
  return Convert(result);
}

output_t optimize_cost(params_t ps, cost_func_t cb, options_t opts) {
  ParamsWrapper x(ps);
  if (!x.is_valid()) return output_t{.stop_reason = -100};

  if (!cb) {
    TINYOPT_LOG("❌ Cost function missing cb = {}", (void *)cb);
    return output_t{.stop_reason = -100};  // Invalid input
  }

  // Wrap C callback into a C++ lambda for the template
  auto cpp_cb = [cb](auto &x) { return cb(x.data()); };
  // Use numerical differentiation to build the accurate function
  auto acc_func1 = [cpp_cb, &opts](auto &x, auto &g) {
    if constexpr (!tinyopt::traits::is_nullptr_v<decltype(g)>) {
      const auto &[res, J] = tinyopt::diff::NumEval(x, cpp_cb, tinyopt::diff::Method::kCentral);
      g = J.transpose() * res;
      return res;
    } else {
      return cpp_cb(x);
    }
  };
  auto acc_func2 = [cpp_cb, &opts](auto &x, auto &g, auto &h) {
    if constexpr (!tinyopt::traits::is_nullptr_v<decltype(g)>) {
      const auto &[res, J] = tinyopt::diff::NumEval(x, cpp_cb, tinyopt::diff::Method::kCentral);
      g = J.transpose() * res;
      h = J.transpose() * J;
      // if (opts.print_grad) TINYOPT_LOG("💡 grad: {}", g);
      // if (opts.print_hessian) TINYOPT_LOG("💡 H: {}", h);
      return res;
    } else {
      return cpp_cb(x);
    }
  };

  // Call the C++ template
  const int is_second_order =
      opts.solver_type == LevenbergMarquardt || opts.solver_type == GaussNewton;
  if (is_second_order) {
    auto result = tinyopt::Optimize(x, acc_func2, Convert(opts));
    return Convert(result);
  } else {
    auto result = tinyopt::Optimize(x, acc_func1, Convert(opts));
    return Convert(result);
  }
}

output_t optimize_cost_grad(params_t ps, cost_grad_func_t cb, options_t opts) {
  ParamsWrapper x(ps);
  if (!x.is_valid()) return output_t{.stop_reason = -100};
  if (!cb) {
    TINYOPT_LOG("❌ Cost function missing cb = {}", (void *)cb);
    return output_t{.stop_reason = -100};  // Invalid input
  }

  // Wrap C callback into a C++ lambda for the template
  auto acc_func = [cb, &opts](auto &x, auto &g) {
    if constexpr (!tinyopt::traits::is_nullptr_v<decltype(g)>) {
      auto r = cb(x.data(), g.data());
      return r;
    } else {
      return cb(x.data(), nullptr);
    }
  };

  tinyopt::Options options = Convert(opts);
  // Call the C++ template
  auto result = tinyopt::Optimize(x, acc_func, Convert(opts));
  return Convert(result);
}
}