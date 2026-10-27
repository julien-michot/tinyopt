// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <tuple>
#include <type_traits>

namespace tinyopt::detail {

template <typename Func, typename... Params>
struct PackedAccumulationAdapter;

/// Extracts the last type of a callable's argument list.
template <typename... Args>
struct LastCallableArgument {
  using Type = void;
};

template <typename First, typename Second, typename... Rest>
struct LastCallableArgument<First, Second, Rest...> {
  using Type = typename LastCallableArgument<Second, Rest...>::Type;
};

template <typename First, typename Second, typename Last>
struct LastCallableArgument<First, Second, Last> {
  using Type = std::remove_cvref_t<Last>;
};

template <typename Signature>
struct CallableHessianArgument : LastCallableArgument<> {};

template <typename Result, typename... Args>
struct CallableHessianArgument<Result(Args...)> : LastCallableArgument<Args...> {};

template <typename Result, typename... Args>
struct CallableHessianArgument<Result(Args...) noexcept> : LastCallableArgument<Args...> {};

template <typename Class, typename Result, typename... Args>
struct CallableHessianArgument<Result (Class::*)(Args...) const> : LastCallableArgument<Args...> {};

template <typename Class, typename Result, typename... Args>
struct CallableHessianArgument<Result (Class::*)(Args...) const noexcept>
    : LastCallableArgument<Args...> {};

template <typename Class, typename Result, typename... Args>
struct CallableHessianArgument<Result (Class::*)(Args...)> : LastCallableArgument<Args...> {};

template <typename Class, typename Result, typename... Args>
struct CallableHessianArgument<Result (Class::*)(Args...) noexcept>
    : LastCallableArgument<Args...> {};

/// Trailing argument type of a callable with a non-templated call operator.
template <typename Func, typename = void>
struct CallableHessian {
  using Type = void;
};

template <typename Func>
struct CallableHessian<Func, std::void_t<decltype(&std::remove_cvref_t<Func>::operator())>>
    : CallableHessianArgument<decltype(&std::remove_cvref_t<Func>::operator())> {};

template <typename Result, typename... Args>
struct CallableHessian<Result (*)(Args...), void> : LastCallableArgument<Args...> {};

template <typename Result, typename... Args>
struct CallableHessian<Result (*)(Args...) noexcept, void> : LastCallableArgument<Args...> {};

template <typename Result, typename... Args>
struct CallableHessian<std::function<Result(Args...)>, void> : LastCallableArgument<Args...> {};

template <typename Func, typename... Params>
struct CallableHessian<PackedAccumulationAdapter<Func, Params...>, void> : CallableHessian<Func> {};

/// Instantiates a (possibly generic) call operator with `Args...` and inspects its trailing
/// argument. Unlike `CallableHessian`, this supports lambdas and functors whose `operator()` is a
/// template, as long as the Hessian parameter itself is not generic:
///   auto loss = [&](const Trajectory &x, auto &grad, BlockTridiagMat<6> &H) { ... };
template <typename Func, typename ArgsTuple, typename = void>
struct CallableHessianProbe {
  using Type = void;
};

template <typename Func, typename... Args>
struct CallableHessianProbe<
    Func, std::tuple<Args...>,
    std::void_t<decltype(&std::remove_cvref_t<Func>::template operator()<Args...>)>>
    : CallableHessianArgument<decltype(&std::remove_cvref_t<Func>::template operator()<Args...>)> {
};

template <typename Func, typename... Params, typename ArgsTuple>
struct CallableHessianProbe<PackedAccumulationAdapter<Func, Params...>, ArgsTuple, void>
    : CallableHessianProbe<Func, ArgsTuple> {};

template <typename Func, typename ArgsTuple, bool Enabled>
struct LazyCallableHessianProbe {
  using Type = void;
};

template <typename Func, typename ArgsTuple>
struct LazyCallableHessianProbe<Func, ArgsTuple, true> : CallableHessianProbe<Func, ArgsTuple> {};

/// True for Eigen column-vector-like types, whose generic `x` can be instantiated with an
/// optimizer gradient of the same shape.
template <typename T, typename = void>
struct IsColumnVectorType : std::false_type {};

template <typename T>
struct IsColumnVectorType<T, std::void_t<decltype(T::ColsAtCompileTime)>>
    : std::bool_constant<T::ColsAtCompileTime == 1> {};

template <typename Func, typename T, typename Gradient, bool Enabled>
struct CallableHessianTypeImpl {
  using Type = typename CallableHessian<Func>::Type;
};

/// Deduces the explicit Hessian argument of a callable with a generic (templated) call operator.
/// The generic call operator is instantiated with the block-Hessian gradient type, which recovers
/// a concrete Hessian argument such as `BlockTridiagMat<N>` or `BlockSparseMatrix<...>` from
/// signatures like:
///   auto loss = [&](const Trajectory &x, auto &grad, BlockTridiagMat<6> &H) { ... };
template <typename Func, typename T, typename Gradient>
struct CallableHessianTypeFallback {
 private:
  // A generic `x` would be instantiated with the gradient type, so the probe is restricted to
  // parameter types that are naturally interchangeable with the gradient (column vectors and
  // scalars). Custom parameter types fall back to the non-generic deduction path.
  static constexpr bool kProbeGradient =
      std::is_same_v<T, Gradient> || std::is_scalar_v<T> || IsColumnVectorType<T>::value;

 public:
  using Type = typename LazyCallableHessianProbe<Func, std::tuple<Gradient>, kProbeGradient>::Type;
};

template <typename Func, typename T, typename Gradient>
struct CallableHessianTypeImpl<Func, T, Gradient, true> {
  using Type = typename CallableHessianTypeFallback<Func, T, Gradient>::Type;
};

/// Explicit Hessian type of `Func`, supporting both concrete and generic call operators.
/// Callables that only take the parameters (cost or residual functions) never expose a Hessian.
template <typename Func, typename T, typename Gradient>
struct CallableHessianType {
 private:
  static constexpr bool kEnabled = std::is_void_v<typename CallableHessian<Func>::Type> &&
                                   !std::is_invocable_v<const Func &, const T &> &&
                                   !std::is_invocable_v<const Func &, const T &, Gradient &>;

 public:
  using Type = typename CallableHessianTypeImpl<Func, T, Gradient, kEnabled>::Type;
};

}  // namespace tinyopt::detail
