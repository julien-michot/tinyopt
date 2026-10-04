// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <limits>
#include <vector>

#include <tinyopt/optimize.h>

namespace tinyopt::c_api_detail {

class UserStopRequested : public std::exception {};

template <typename Scalar, typename Params, typename Callback>
class CostGradientAccumulator {
 public:
  CostGradientAccumulator(const Params &initial_params, Callback callback, void *user_data)
      : callback_(callback),
        user_data_(user_data),
        params_plus_(initial_params),
        params_minus_(initial_params),
        step_(Vector<Scalar, Params::Dims>::Zero(initial_params.dims())) {}

  template <typename Gradient>
  Cost operator()(const Params &params, Gradient &gradient) const {
    const Scalar cost = Evaluate(params);
    if constexpr (!traits::is_nullptr_v<Gradient>) {
      for (Index i = 0; i < params.dims(); ++i) {
        std::copy(params.values.begin(), params.values.end(), params_plus_.values.begin());
        std::copy(params.values.begin(), params.values.end(), params_minus_.values.begin());
        step_.setZero();
        step_[i] = FloatEpsilon<Scalar>() * std::max(Scalar(1), std::abs(params.values[i]));
        traits::params_trait<Params>::PlusEq(params_plus_, step_);
        traits::params_trait<Params>::PlusEq(params_minus_, -step_);
        gradient[i] = (Evaluate(params_plus_) - Evaluate(params_minus_)) / (Scalar(2) * step_[i]);
      }
    }
    return Cost(cost, 1);
  }

 private:
  Scalar Evaluate(const Params &params) const {
    Scalar cost = 0;
    if (callback_(params.values.data(), static_cast<int>(params.dims()), &cost, user_data_) != 0)
      throw UserStopRequested{};
    return cost;
  }

  Callback callback_;
  void *user_data_;
  mutable Params params_plus_;
  mutable Params params_minus_;
  mutable Vector<Scalar, Params::Dims> step_;
};

template <typename Scalar, typename Params, typename Callback>
struct GradientCallback {
  Callback callback;
  void *user_data;

  template <typename Gradient>
  Scalar operator()(const Params &params, Gradient &gradient) const {
    Scalar cost = 0;
    Scalar *gradient_data = nullptr;
    if constexpr (!traits::is_nullptr_v<Gradient>) gradient_data = gradient.data();
    if (callback(params.values.data(), static_cast<int>(params.dims()), &cost, gradient_data,
                 user_data) != 0)
      throw UserStopRequested{};
    return cost;
  }
};

template <typename Scalar, typename Params, typename Callback>
struct HessianCallback {
  Callback callback;
  void *user_data;

  template <typename Gradient, typename Hessian>
  Scalar operator()(const Params &params, Gradient &gradient, Hessian &hessian) const {
    Scalar cost = 0;
    Scalar *gradient_data = nullptr;
    Scalar *hessian_data = nullptr;
    if constexpr (!traits::is_nullptr_v<Gradient>) gradient_data = gradient.data();
    if constexpr (!traits::is_nullptr_v<Hessian>) hessian_data = hessian.data();
    if (callback(params.values.data(), static_cast<int>(params.dims()), &cost, gradient_data,
                 hessian_data, user_data) != 0)
      throw UserStopRequested{};
    return cost;
  }
};

template <typename Scalar, typename Params, typename Callback>
class ResidualAccumulator {
 public:
  ResidualAccumulator(const Params &initial_params, int residual_dims, Callback callback,
                      void *user_data)
      : residual_dims_(residual_dims),
        callback_(callback),
        user_data_(user_data),
        residuals_(static_cast<std::size_t>(residual_dims)),
        residuals_plus_(static_cast<std::size_t>(residual_dims)),
        residuals_minus_(static_cast<std::size_t>(residual_dims)),
        jacobian_(static_cast<std::size_t>(residual_dims) *
                  static_cast<std::size_t>(initial_params.dims())),
        params_plus_(initial_params),
        params_minus_(initial_params),
        step_(initial_params.dims()) {}

  template <typename Gradient, typename Hessian>
  Cost operator()(const Params &params, Gradient &gradient, Hessian &hessian) const {
    constexpr bool HasGradient = !traits::is_nullptr_v<Gradient>;
    constexpr bool HasHessian = !traits::is_nullptr_v<Hessian>;
    const auto param_dims = params.dims();
    // Cost-only evaluations may pass an unsized dummy dynamic Hessian: nothing to accumulate.
    bool has_system = HasGradient;
    if constexpr (HasHessian) has_system = has_system || hessian.size() > 0;

    Scalar *jacobian_data = has_system ? jacobian_.data() : nullptr;
    jacobian_data = Evaluate(params, residuals_, jacobian_data);
    if (has_system) {
      if (jacobian_data == nullptr) {
        used_numerical_differentiation_ = true;
        EstimateJacobian(params, param_dims);
      }
      if constexpr (HasGradient) gradient.setZero();
      if constexpr (HasHessian) hessian.setZero();
      Accumulate(params, gradient, hessian, param_dims);
    }

    Scalar squared_norm = 0;
    for (const Scalar residual : residuals_) squared_norm += residual * residual;
    return Cost(std::sqrt(squared_norm), residual_dims_);
  }

  bool UsedNumericalDifferentiation() const { return used_numerical_differentiation_; }

 private:
  Scalar *Evaluate(const Params &params, std::vector<Scalar> &residuals, Scalar *jacobian) const {
    if (callback_(params.values.data(), static_cast<int>(params.dims()), residuals.data(),
                  &jacobian, residual_dims_, user_data_) != 0)
      throw UserStopRequested{};
    return jacobian;
  }

  void EstimateJacobian(const Params &params, Index param_dims) const {
    const Scalar base_step = FloatEpsilon<Scalar>();
    for (Index column = 0; column < param_dims; ++column) {
      std::copy(params.values.begin(), params.values.end(), params_plus_.values.begin());
      std::copy(params.values.begin(), params.values.end(), params_minus_.values.begin());
      step_.setZero();
      step_[column] = base_step * std::max(Scalar(1), std::abs(params.values[column]));
      traits::params_trait<Params>::PlusEq(params_plus_, step_);
      traits::params_trait<Params>::PlusEq(params_minus_, -step_);
      Evaluate(params_plus_, residuals_plus_, nullptr);
      Evaluate(params_minus_, residuals_minus_, nullptr);
      const Scalar denominator = Scalar(2) * step_[column];
      for (int row = 0; row < residual_dims_; ++row) {
        jacobian_[static_cast<std::size_t>(row) * static_cast<std::size_t>(param_dims) +
                  static_cast<std::size_t>(column)] =
            (residuals_plus_[static_cast<std::size_t>(row)] -
             residuals_minus_[static_cast<std::size_t>(row)]) /
            denominator;
      }
    }
  }

  template <typename Gradient, typename Hessian>
  void Accumulate(const Params &, Gradient &gradient, Hessian &hessian, Index param_dims) const {
    for (int row = 0; row < residual_dims_; ++row) {
      const Scalar residual = residuals_[static_cast<std::size_t>(row)];
      const std::size_t row_offset =
          static_cast<std::size_t>(row) * static_cast<std::size_t>(param_dims);
      for (Index col = 0; col < param_dims; ++col) {
        const Scalar jac_col = jacobian_[row_offset + static_cast<std::size_t>(col)];
        if constexpr (!traits::is_nullptr_v<Gradient>) gradient[col] += jac_col * residual;
        if constexpr (!traits::is_nullptr_v<Hessian>) {
          for (Index other = 0; other < param_dims; ++other) {
            const Scalar jac_other = jacobian_[row_offset + static_cast<std::size_t>(other)];
            hessian(col, other) += jac_col * jac_other;
          }
        }
      }
    }
  }

  int residual_dims_;
  Callback callback_;
  void *user_data_;
  mutable std::vector<Scalar> residuals_;
  mutable std::vector<Scalar> residuals_plus_;
  mutable std::vector<Scalar> residuals_minus_;
  mutable std::vector<Scalar> jacobian_;
  mutable Params params_plus_;
  mutable Params params_minus_;
  mutable Vector<Scalar, Params::Dims> step_;
  mutable bool used_numerical_differentiation_ = false;
};

template <typename Params, typename Problem>
Summary OptimizeProblem(Params &params, const Problem &problem, const Options &options) {
  using Scalar = typename Params::Scalar;
  switch (problem.type) {
    case TINYOPT_EVAL_COST_ONLY: {
      if (problem.fn.cost == nullptr) throw std::invalid_argument("Missing cost callback");
      Options cost_options = options;
      if (cost_options.solver_type == Options::LevenbergMarquardt)
        cost_options.solver_type = Options::Solver::BFGS;
      CostGradientAccumulator<Scalar, Params, decltype(problem.fn.cost)> accumulator(
          params, problem.fn.cost, problem.user_data);
      return Optimize(params, accumulator, cost_options);
    }
    case TINYOPT_EVAL_GRADIENT: {
      if (problem.fn.acc_grad == nullptr)
        throw std::invalid_argument("Missing gradient accumulation callback");
      return Optimize(params,
                      GradientCallback<Scalar, Params, decltype(problem.fn.acc_grad)>{
                          problem.fn.acc_grad, problem.user_data},
                      options);
    }
    case TINYOPT_EVAL_HESSIAN: {
      if (problem.fn.acc_hessian == nullptr)
        throw std::invalid_argument("Missing Hessian accumulation callback");
      return Optimize(params,
                      HessianCallback<Scalar, Params, decltype(problem.fn.acc_hessian)>{
                          problem.fn.acc_hessian, problem.user_data},
                      options);
    }
    case TINYOPT_EVAL_RESIDUALS: {
      if (problem.fn.residuals == nullptr || problem.num_residuals <= 0)
        throw std::invalid_argument("Invalid residual callback or count");
      const auto param_dims = static_cast<std::size_t>(params.dims());
      if (static_cast<std::size_t>(problem.num_residuals) >
          std::numeric_limits<std::size_t>::max() / param_dims)
        throw std::invalid_argument("Residual Jacobian size overflow");
      ResidualAccumulator<Scalar, Params, decltype(problem.fn.residuals)> accumulator(
          params, problem.num_residuals, problem.fn.residuals, problem.user_data);
      auto result = Optimize(params, accumulator, options);
      result.num_diff_used = result.num_diff_used || accumulator.UsedNumericalDifferentiation();
      return result;
    }
  }
  throw std::invalid_argument("Unknown C evaluation type");
}

}  // namespace tinyopt::c_api_detail