// Copyright (C) 2025 Julien Michot. All Rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cstddef>
#include <type_traits>

#include <tinyopt/traits.h>

namespace tinyopt {

/**
 * @brief A generic parameter structure that encapsulates a variable `x` and an optional manifold.
 *
 * This struct allows for parameter manipulation, especially in optimization contexts, where
 * parameters might live on non-Euclidean manifolds. It supports both statically and
 * dynamically sized parameter vectors.
 *
 * @tparam T The type of the parameter `x`.
 * @tparam _Dims The dimension of the parameter vector. Can be `Dynamic` for runtime sizing.
 * @tparam Manifold A callable object (e.g., lambda, function pointer) that defines the manifold
 * operation. If `std::nullptr_t`, standard addition is used for updates.
 */
template <typename T, int _Dims = Dynamic, typename Manifold = std::nullptr_t>
struct Params {
  using X_t = T;
  using Scalar = typename traits::params_trait<T>::Scalar;
  static constexpr int Dims = _Dims;

  /**
   * @brief Constructor for statically sized parameters (Dims != Dynamic).
   *
   * @param _x Reference to the parameter variable.
   * @param manifold Optional manifold function. Defaults to an empty manifold.
   */
  template <int D = Dims, std::enable_if_t<D != Dynamic, int> = 0>
  explicit Params(X_t &_x, const Manifold &manifold = {}) : manifold_{manifold}, x{_x} {}

  /**
   * @brief Constructor for dynamically sized parameters (Dims == Dynamic).
   *
   * @param _x Reference to the parameter variable.
   * @param _dims The runtime dimension of the parameter.
   * @param manifold Optional manifold function. Defaults to an empty manifold.
   */
  template <int D = Dims, std::enable_if_t<D == Dynamic, int> = 0>
  explicit Params(X_t &_x, int _dims, const Manifold &manifold = {})
      : dims_{_dims}, manifold_{manifold}, x{_x} {}
  /**
   * @brief Copy constructor.
   *
   * No deep copy here.
   *
   * @param other The `Params` object to copy from.
   */
  Params(const Params &other) : dims_(other.dims_), manifold_(other.manifold_), x(other.x) {}

  /**
   * @brief Copy assignment operator.
   *
   * Assigns the contents of another `Params` object to this object, performing a deep copy.
   *
   * @param other The `Params` object to copy from.
   * @return A reference to this `Params` object.
   */
  Params &operator=(const Params &other) {
    if (this != &other) {
      const_cast<int &>(dims_) = other.dims_;
      const_cast<Manifold &>(manifold_) = other.manifold_;
      x = other.x;
    }
    return *this;
  }

  /**
   * @brief Returns the dimension of the parameter.
   *
   * @param p The `Params` object.
   * @return The dimension of the parameter.
   */
  static auto dims(const Params &p) { return Dims == Dynamic ? p.dims_ : Dims; }

  /**
   * @brief Updates the parameter `x` using the provided delta.
   *
   * If a manifold is provided, it is used for the update; otherwise, standard addition is used.
   *
   * @param delta The update delta.
   */
  void operator+=(const auto &delta) {
    if constexpr (std::is_same_v<Manifold, std::nullptr_t>) {
      traits::params_trait<X_t>::pluseq(x, delta);
    } else if (std::is_invocable_v<Manifold, X_t &, const decltype(delta) &>) {
      manifold_(x, delta);
    } else {
      traits::params_trait<X_t>::pluseq(x, delta);
    }
  }

  /**
   * @brief The runtime dimension of the parameter (only used when Dims == Dynamic).
   */
  const int dims_ = Dims;
  /**
   * @brief The manifold function, if provided.
   */
  const Manifold &manifold_;
  /**
   * @brief Reference to the parameter variable.
   */
  X_t &x;
};

/// Convenient function that run callback(x.x) if x is a Params class, else run callback(x)
template <typename X_t, typename Callback_t>
auto MaybeParamsRun(const X_t &x, const Callback_t &callback) {
  if constexpr (traits::is_params_class_v<X_t>)
    return callback(x.x);
  else
    return callback(x);
}

///
template <typename T2, typename X_t>
auto MaybeParamsCast(const X_t &x) {
  if constexpr (traits::is_params_class_v<X_t>)
    return traits::params_trait<typename X_t::X_t>::template cast<T2>(x.x);
  else
    return traits::params_trait<X_t>::template cast<T2>(x);
}

namespace traits {

template <typename T, int Dims, typename Manifold>
struct is_params_class<tinyopt::Params<T, Dims, Manifold>> : std::true_type {};

template <typename T, int Dims, typename Manifold>
struct params_class<tinyopt::Params<T, Dims, Manifold>> {
  static constexpr bool is_ok = true;
  using Scalar = T;
};

// Trait specialization for scalar (float, double)
template <typename X_t, int _Dims, typename Manifold>
struct params_trait<tinyopt::Params<X_t, _Dims, Manifold>> {
  using T = tinyopt::Params<X_t, _Dims, Manifold>;
  using Scalar = typename T::Scalar;  // The scalar type
  static constexpr int Dims = _Dims;  // Compile-time parameters dimensions

  // Execution-time parameters dimensions
  static int dims(const T &p) { return Dims == Dynamic ? p.dims_ : Dims; }

  // Cast to a new type, only needed when using automatic differentiation
  template <typename T2>
  static auto cast(const T &p) {
    return traits::params_trait<X_t>::template cast<T2>(p.x);
  }

  // Define update / manifold
  static void pluseq(T &p, const auto &delta) { p += delta; }
};

}  // namespace traits

/**
 * @brief Creates a `Params` object for statically sized parameters.
 *
 * @tparam Dims The static dimension of the parameter.
 * @tparam T The type of the parameter.
 * @tparam Manifold The type of the manifold function.
 * @param x Reference to the parameter variable.
 * @param manifold Optional manifold function. Defaults to an empty manifold.
 * @return A `Params` object.
 * Usage example:
 * @code
 *  Mat3 R = Mat3::Identity();
 *  auto manifold = [](auto &R, const Vec3 &w) { R *= hat(w); };
 *  auto x = CreateParams<3>(R, manifold);
 * @endcode
 */
template <int Dims, typename T, typename Manifold>
auto CreateParams(T &x, const Manifold &manifold = {}) {
  using ParamsType = Params<T, Dims, Manifold>;
  if constexpr (std::is_same_v<Manifold, std::nullptr_t>)
    return ParamsType(x);
  else {
    return ParamsType(x, manifold);
  }
}

/**
 * @brief Creates a `Params` object for dynamically sized parameters.
 *
 * @tparam T The type of the parameter.
 * @tparam Manifold The type of the manifold function.
 * @param x Reference to the parameter variable.
 * @param dims The runtime dimension of the parameter.
 * @param manifold Optional manifold function. Defaults to an empty manifold.
 * @return A `Params` object.
 * Usage example:
 * @code
 *  Mat3 R = Mat3::Identity();
 *  auto manifold = [](auto &R, const Vec3 &w) { R *= hat(w); };
 *  auto x = CreateParams(R, 3, manifold);
 * @endcode
 */
template <typename T, typename Manifold>
auto CreateParams(T &x, int dims, const Manifold &manifold = {}) {
  using ParamsType = Params<T, Dynamic, Manifold>;
  if constexpr (std::is_same_v<Manifold, std::nullptr_t>)
    return ParamsType(x, dims);
  else {
    return ParamsType(x, dims, manifold);
  }
}

}  // namespace tinyopt
