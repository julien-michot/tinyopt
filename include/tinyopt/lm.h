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

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <ostream>
#include <type_traits>
#include <vector>

#include "jet.h"    // Import Ceres'Jet
#include "log.h"    // Defines TINYOPT_FORMAT and toString
#include "math.h"   // Defines Matrix and Vector
#include "traits.h" // Defines parameters_size_v

namespace tinyopt::lm {

/***
 *  @brief LM Optimization options
 *
 ***/
struct Options {
  double damping_init = 1e-4; // Initial damping factor
  std::array<double, 2> damping_range{
      {1e-9, 1e9}}; // Min and max damping values
  bool ldlt = true; // If not, will use JtJ.inverse()
  bool JtJ_is_full =
      true; // Specify if JtJ is only Upper triangularly or fully filled
  bool export_JtJ = true; // Save and return the last JtJ as part of the output
  // Stops criteria
  uint16_t num_iters = 100;        // Maximum number of iterations
  float min_delta_norm2 = 1e-16;   // Minimum delta (step) squared norm
  float min_grad_norm2 = 1e-16;    // Minimum gradient squared norm
  uint8_t max_total_failures = 0;  // Overall max failures to decrease error
  uint8_t max_consec_failures = 3; // Max consecutive failures to decrease error
  // Logging
  bool log_x = true;             // Log the value of 'x'
  std::ostream &oss = std::cout; // Stream used for logging
};

/***
 *  @brief Struct containing optimization results
 *
 ***/
template <typename JtJ_t> struct Output {
  enum StopReason : uint8_t {
    kMaxIters = 0,   // Reached maximum number of iterations (success)
    kMinDeltaNorm,   // Reached minimal delta norm (success)
    kMinGradNorm,    // Reached minimal gradient (success)
    kMaxFails,       // Failed to decrease error too many times (success)
    kMaxConsecFails, // Failed to decrease error consecutively too many times (success)
    // Failures
    kSystemHasNaNs,  // Residuals or Jacobians have NaNs
    kSolverFailed,   // Failed to solve the normal equations (inverse JtJ)
    kNoResiduals     // The system has no residuals
  };

  // Last valid step results
  float last_err2 = std::numeric_limits<float>::max();

  StopReason stop_reason = StopReason::kMaxIters;
  bool Succeeded() const {
    return stop_reason != StopReason::kSystemHasNaNs &&
           stop_reason != StopReason::kSolverFailed &&
           stop_reason != StopReason::kNoResiduals;
  }
  bool Converged() const {
    return stop_reason == StopReason::kMinDeltaNorm ||
           stop_reason == StopReason::kMinGradNorm;
  }

  uint16_t num_residuals = 0; // Final number of residuals
  uint16_t num_iters = 0;     // Final number of iterations
  uint8_t num_failures = 0;   // Final number of failures to decrease the error
  uint8_t num_consec_failures =
      0; // Final number of the last consecutive failures to decrease the error
  JtJ_t last_JtJ; // Final JtJ, including damping

  // Per iteration results
  std::vector<float> errs2; // Mean squared accumulated errors of all iterations
  std::vector<float> deltas2;  // Step sizes of all iterations
  std::vector<bool> successes; // Step acceptation status for all iterations
};

/***
 *  @brief Minimize a loss function @arg acc using the Levenberg-Marquardt
 *  minimization algorithm.
 *
 ***/
template <typename ParametersType, typename ResidualsFunc>
inline auto LM(ParametersType &X, ResidualsFunc &acc,
               const Options &options = Options{}) {
  using std::sqrt;
  using ptrait = traits::params_trait<ParametersType>;

  using Scalar = ptrait::Scalar;
  constexpr int Size = ptrait::Dims;

  using JtJ_t = Matrix<Scalar, Size, Size>;
  using OutputType = Output<JtJ_t>;
  bool already_rolled_true = true;
  const int size = ptrait::dims(X); // System size (dynamic)
  const uint8_t max_tries =
      options.max_consec_failures > 0
          ? std::max<uint8_t>(1, options.max_total_failures)
          : 255;
  Matrix<Scalar, Size, 1> Jt_res(size, 1);
  auto X_last_good = X;
  double lambda = options.damping_init;
  OutputType out;
  out.errs2.reserve(out.num_iters + 2);
  out.deltas2.emplace_back(out.num_iters + 2);
  out.successes.emplace_back(out.num_iters + 2);
  if (options.export_JtJ)
    out.last_JtJ = JtJ_t::Zero(size, size);
  JtJ_t JtJ(size, size);
  Matrix<Scalar, Size, 1> dX;
  for (; out.num_iters < options.num_iters + 1 /*+1 to potentially roll-back*/;
       ++out.num_iters) {
    JtJ.setZero();
    Jt_res.setZero();
    const auto &output = acc(X, JtJ, Jt_res);
    double err; // accumulated error (for monotony check and logging)
    int nerr;   // number of residuals (optional, for logging)
    if constexpr (std::is_scalar_v<std::remove_reference_t<decltype(output)>>) {
      err = output;
      nerr = 1;
    } else {
      err = std::get<0>(output);
      nerr = std::get<1>(output);
    }
    const bool skip_solver = nerr == 0;
    out.num_residuals = nerr;
    if (nerr == 0) {
      out.errs2.emplace_back(0);
      out.deltas2.emplace_back(0);
      out.successes.emplace_back(false);
      options.oss << TINYOPT_FORMAT("❌ #{}: No residuals, stopping",
                                    out.num_iters)
                  << std::endl;
      // Can break only if first time, otherwise better count it as failure
      if (out.num_iters == 0) {
        out.stop_reason = OutputType::StopReason::kNoResiduals;
        break;
      }
    }

    // Damping
    for (int i = 0; i < size; ++i)
      JtJ(i, i) *= (1.0 + lambda);

    dX.setZero();
    bool solver_failed = skip_solver;
    bool system_has_nans = false;
    for (; !skip_solver && out.num_consec_failures <= max_tries;) {
      // Solver linear system
      if (options.ldlt) {
        const auto chol =
            Eigen::SelfAdjointView<const JtJ_t, Eigen::Upper>(JtJ).ldlt();
        if (chol.isPositive()) {
          dX = -chol.solve(Jt_res);
          solver_failed = false;
          break;
        } else {
          solver_failed = true;
          dX.setZero();
        }
      } else { // Use Eigen's default inverse
        // Fill the lower part of JtJ then inverse it
        if (!options.JtJ_is_full)
          JtJ.template triangularView<Eigen::Lower>() =
              JtJ.template triangularView<Eigen::Upper>().transpose();
        dX = -JtJ.inverse() * Jt_res;
        solver_failed = false;
        break;
      }
      // Sover failed -> re-do the damping
      double lambda2 =
          std::min(options.damping_range[1],
                   std::max(options.damping_range[0], lambda * 10));
      const double s = (1.0 + lambda2) / (1.0 * lambda);
      options.oss << TINYOPT_FORMAT(
                         "❌ #{}: Cholesky Failed, redamping to λ:{:.2e}",
                         out.num_iters, s)
                  << std::endl;
      for (int i = 0; i < size; ++i)
        JtJ(i, i) *= s;
      lambda = lambda2;
      out.num_consec_failures++;
      out.num_failures++;
    }

    // Check the displacement magnitude
    const double dX_norm2 = solver_failed ? 0 : dX.squaredNorm();
    const double Jt_res_norm2 =
        options.min_grad_norm2 == 0.0f ? 0 : Jt_res.squaredNorm();
    if (std::isnan(dX_norm2)) {
      solver_failed = true;
      options.oss << TINYOPT_FORMAT("❌ Failure, dX = \n{}",
                                    toString(dX.template cast<float>()))
                  << std::endl;
      options.oss << TINYOPT_FORMAT("JtJ = \n{}", toString(JtJ)) << std::endl;
      options.oss << TINYOPT_FORMAT("Jt*res = \n{}", toString(Jt_res))
                  << std::endl;
      system_has_nans = true;
      break;
    }

    const double derr = err - out.last_err2;
    // Save history of errors and deltas
    out.errs2.emplace_back(err);
    out.deltas2.emplace_back(dX_norm2);
    // Check step quality
    if (derr < 0.0 && !solver_failed) { /* GOOD Step */
      out.successes.emplace_back(true);
      if (out.num_iters > 0)
        X_last_good = X;
      // Move X by dX
      ptrait::pluseq(X, dX);
      // Save results
      out.last_err2 = err;
      if (options.export_JtJ) {
        out.last_JtJ =
            JtJ; // TODO actually better store the one right after a success
        for (int i = 0; i < Size; ++i)
          out.last_JtJ(i, i) = JtJ(i, i) / (1.0f + lambda);
      }
      already_rolled_true = false;
      out.num_consec_failures = 0;
      if (options.log_x) {
        options.oss << TINYOPT_FORMAT(
                           "✅ #{}: X:{} |δX|:{:.2e} λ:{:.2e} ⎡σ⎤:{:.4f} "
                           "ε²:{:.5f} n:{} dε²:{:.3e} ∇ε²:{:.3e}",
                           out.num_iters, ptrait::toString(X), sqrt(dX_norm2), lambda,
                           sqrt(InvCov(JtJ).maxCoeff()), err, nerr, derr,
                           Jt_res_norm2)
                    << std::endl;
      } else {
        options.oss << TINYOPT_FORMAT("✅ #{}: |δX|:{:.2e} λ:{:.2e} ε²:{:.5f} "
                                      "n:{} dε²:{:.3e} ∇ε²:{:.3e}",
                                      out.num_iters, std::sqrt(dX_norm2),
                                      lambda, err, nerr, derr, Jt_res_norm2)
                    << std::endl;
      }
      lambda = std::min(options.damping_range[1],
                        std::max(options.damping_range[0], lambda / 3.0));
    } else { /* BAD Step */
      out.successes.emplace_back(false);
      if (options.log_x) {
        options.oss << TINYOPT_FORMAT(
                           "❌ #{}: X:{} |δX|:{:.2e} λ:{:.2e} ε²:{:.5f} n:{} "
                           "dε²:{:.3e} ∇ε²:{:.3e}",
                           out.num_iters, ptrait::toString(X), sqrt(dX_norm2), lambda,
                           err, nerr, derr, Jt_res_norm2)
                    << std::endl;
      } else {
        options.oss << TINYOPT_FORMAT("❌ #{}: |δX|:{:.2e} λ:{:.2e} ε²:{:.5f} "
                                      "n:{} dε²:{:.3e} ∇ε²:{:.3e}",
                                      out.num_iters, std::sqrt(dX_norm2),
                                      lambda, err, nerr, derr, Jt_res_norm2)
                    << std::endl;
      }
      if (!already_rolled_true) {
        X = X_last_good; // roll back by copy
        already_rolled_true = true;
      }
      out.num_failures++;
      out.num_consec_failures++;
      if (options.max_consec_failures > 0 &&
          out.num_consec_failures >= options.max_consec_failures) {
        out.stop_reason = OutputType::StopReason::kMaxConsecFails;
        break;
      }
      if (options.max_total_failures > 0 &&
          out.num_failures >= options.max_total_failures) {
        out.stop_reason = OutputType::StopReason::kMaxFails;
        break;
      }
      lambda = std::min(options.damping_range[1],
                        std::max(options.damping_range[0], lambda * 2));
      // TODO don't rebuild if no rollback!
    }
    if (system_has_nans) {
      out.stop_reason = OutputType::StopReason::kSystemHasNaNs;
      break;
    } else if (solver_failed) {
      out.stop_reason = OutputType::StopReason::kSolverFailed;
      break;
    }
    if (options.min_delta_norm2 > 0 && dX_norm2 < options.min_delta_norm2) {
      out.stop_reason = OutputType::StopReason::kMinDeltaNorm;
      break;
    }
    if (options.min_grad_norm2 > 0 && Jt_res_norm2 < options.min_grad_norm2) {
      out.stop_reason = OutputType::StopReason::kMinGradNorm;
      break;
    }
  }
  return out;
}

/***
 *  @brief Minimize a loss function @arg residuals using the Levenberg-Marquardt
 *  minimization algorithm and automatic differentiation (Jet) on the loss
 *  function.
 *
 ***/
template <typename ParametersType, typename ResidualsFunc>
inline auto LMJet(ParametersType &X, ResidualsFunc &residuals,
                  const Options &options = Options{}) {

  using ptrait = traits::params_trait<ParametersType>;
  using Scalar = ptrait::Scalar;
  constexpr int Size = ptrait::Dims;
  constexpr bool is_userdef_type = !std::is_floating_point_v<ParametersType>  && !traits::is_eigen_matrix_or_array_v<ParametersType>;

  const int size = ptrait::dims(X);
  // Construct the Jet
  using Jet = Jet<Scalar, Size>;
  // XJetType is either of {Jet, Vector<Jet, N> or ParametersType::cast<Jet>()}
  using XJetType = std::conditional_t<std::is_floating_point_v<ParametersType>, Jet,
                                      std::conditional_t<traits::is_eigen_matrix_or_array_v<ParametersType>,
                                        Vector<Jet, Size>, decltype(ptrait::template cast<Jet>(X))>>;
  // DXJetType is either of {nullptr, Vector<Jet, N>}
  using DXJetType = std::conditional_t<is_userdef_type, Vector<Jet, Size>, std::nullptr_t>;
  XJetType x_jet;
  DXJetType dx_jet; // only for user defined X type

  // Copy X to Jet values
  if constexpr (is_userdef_type) { // X is user defined object
    dx_jet = DXJetType::Zero(size);
    for (int i = 0; i < size; ++i) {
      dx_jet[i].v[i] = 1;
    }
    // dx_jet is constant
  } else if constexpr (std::is_floating_point_v<ParametersType>) { // X is scalar
    x_jet = XJetType(size);
    x_jet.v[0] = 1;
  } else { // X is a Vector
    x_jet = XJetType(size);
    for (int i = 0; i < size; ++i) {
      x_jet[i].v[i] = 1;
    }
  }

  auto acc = [&](const auto &x, auto &JtJ, auto &Jt_res) {

    // Update jet with latest 'x' values
    if constexpr (is_userdef_type) { // X is user defined object
      x_jet = ptrait::template cast<Jet>(X); // Cast X to a Jet type
      using ptrait_jet = traits::params_trait<XJetType>;
      ptrait_jet::pluseq(x_jet, dx_jet);
    } else if constexpr (std::is_floating_point_v<ParametersType>) { // X is scalar
      x_jet.a = x;
    } else { // X is a Vector
      for (int i = 0; i < size; ++i) {
        x_jet[i].a = x[i];
      }
    }

    // Retrieve the residuals
    const auto res = residuals(x_jet);
    using ResType =
        typename std::remove_const_t<std::remove_reference_t<decltype(res)>>;

    if constexpr (!traits::is_eigen_matrix_or_array_v<ResType> &&
                  std::is_floating_point_v<ParametersType>) {
      // Update JtJ and Jt*err
      const auto &J = res.v;
      JtJ(0, 0) = J[0] * J[0];
      Jt_res[0] = J[0] * res.a; // gradient
      // Return both the squared error and the number of residuals
      return std::make_pair(res.a * res.a, 1);
    } else { // Extract jacobian (TODO speed this up)
      constexpr int ResSize = traits::params_trait<ResType>::Dims;
      int res_size = ResSize; // System size (dynamic)
      if constexpr (ResSize != 1 && !std::is_floating_point_v<
                                        std::remove_reference_t<decltype(res)>>)
        res_size = res.size();

      // TODO avoid this copy
      Matrix<Scalar, ResSize, Size> J(res_size, size);
      for (int i = 0; i < res_size; ++i) {
        if constexpr (traits::is_eigen_matrix_or_array_v<ResType>)
          J.row(i) = res[i].v;
        else
          J.row(i) = res.v;
      }
      Vector<Scalar, ResSize> res_f(res.rows());
      for (int i = 0; i < res.rows(); ++i) {
        if constexpr (traits::is_eigen_matrix_or_array_v<ResType>)
          res_f[i] = res[i].a;
        else
          res_f[i] = res.a;
      }
      // Update JtJ and Jt*err
      JtJ = J.transpose() * J;
      Jt_res = J.transpose() * res_f;
      // Returns the squared residuals norm
      return std::make_pair(res_f.squaredNorm(), res_size);
    }
  };

  return LM(X, acc, options);
}

/***
 *  @brief Minimize a loss function @arg acc using the Levenberg-Marquardt
 *  minimization algorithm.
 *
 ***/
template <typename ParametersType, typename ResidualsFunc>
inline auto Optimize(ParametersType &x, ResidualsFunc &func,
                     const Options &options = Options{}) {
  if constexpr (std::is_invocable_v<ResidualsFunc, const ParametersType &>) {
    return LMJet(x, func, options);
  } else {
    return LM(x, func, options);
  }
}

} // namespace tinyopt::lm
