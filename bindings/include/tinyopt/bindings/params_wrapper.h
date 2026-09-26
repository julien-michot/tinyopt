#pragma once

#include <Eigen/Core>

#include <functional>

#include <tinyopt/types.h>

namespace tinyopt::bindings {

// Shared parameter wrapper used by bindings (Python/C/JS).
//
// Contract:
// - Scalar is always double for the bindings layer.
// - Stores a copy of the current parameter vector.
// - Provides dims()/data() so tinyopt's params_trait can treat it like a vector.
// - operator+= applies an optional manifold "plus" operation when provided.
struct ParamsWrapper {
  using Scalar = double;

  ParamsWrapper() = default;

  // Create from a raw pointer (typically coming from a binding layer).
  // Copies into an owning VecX.
  ParamsWrapper(const double* ptr, size_t n)
      : x(Eigen::Map<const VecX>(ptr, static_cast<Eigen::Index>(n))) {}

  int dims() const { return static_cast<int>(x.size()); }

  double* data() { return x.data(); }
  const double* data() const { return x.data(); }

  ParamsWrapper& operator+=(const VecX& delta) {
    if (plus_manifold) {
      x = plus_manifold(x, delta);
    } else {
      x += delta;
    }
    return *this;
  }

  VecX x;

  // Optional manifold addition (x, delta) -> x_plus.
  // Bindings can install a lambda that calls back into their runtime.
  std::function<VecX(const VecX&, const VecX&)> plus_manifold;
};

}  // namespace tinyopt::bindings
