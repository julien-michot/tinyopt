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

#include <tinyopt/gn.h>
#include <functional>

namespace tinyopt::lm {

/***
 *  @brief LM Optimization options
 *
 ***/
struct Options : tinyopt::gn::Options {
  Options() : tinyopt::gn::Options() {
    this->max_total_failures = 0;   // Overall max failures to decrease error
    this->max_consec_failures = 3;  // Max consecutive failures to decrease error
  }
  double damping_init = 1e-4;  // Initial damping factor
  // Min and max damping values
  std::array<double, 2> damping_range{{1e-9, 1e9}};
};

template <typename JtJ_t>
using Output = tinyopt::gn::Output<JtJ_t>;

/***
 *  @brief Minimize a loss function @arg acc using the Levenberg-Marquardt minimization algorithm.
 *
 ***/
template <typename ParametersType, typename ResidualsFunc>
inline auto LM(ParametersType &X, ResidualsFunc &&acc, const Options &options = Options{}) {
  using std::sqrt;
  using ptrait = traits::params_trait<ParametersType>;

  using Scalar = ptrait::Scalar;
  constexpr int Size = ptrait::Dims;

  int size = Size; // Dynamic size
  if constexpr (Size == Eigen::Dynamic) size = ptrait::dims(X);

  using JtJ_t = Matrix<Scalar, Size, Size>;
  using OutputType = Output<JtJ_t>;
  bool already_rolled_true = true;
  const uint8_t max_tries =
      options.max_consec_failures > 0 ? std::max<uint8_t>(1, options.max_total_failures) : 255;
  auto X_last_good = X;
  double lambda = options.damping_init;
  OutputType out;
  out.errs2.reserve(out.num_iters + 2);
  out.deltas2.emplace_back(out.num_iters + 2);
  out.successes.emplace_back(out.num_iters + 2);
  if (options.export_JtJ) out.last_JtJ = JtJ_t::Zero(size, size);
  JtJ_t JtJ(size, size);
  Matrix<Scalar, Size, 1> Jt_res(size, 1);
  Matrix<Scalar, Size, 1> dX(size, 1);
  for (; out.num_iters < options.num_iters + 1 /*+1 to potentially roll-back*/; ++out.num_iters) {
    JtJ.setZero();
    Jt_res.setZero();;
    const auto &output = acc(X, JtJ, Jt_res);
    double err;  // accumulated error (for monotony check and logging)
    int nerr;    // number of residuals (optional, for logging)

    using ResOutputType = std::remove_reference_t<decltype(output)>;
    if constexpr (std::is_scalar_v<ResOutputType>) {
      err = output;
      nerr = 1;
    } else { // output must be a pair/tuple
      err = std::get<0>(output);
      nerr = std::get<1>(output);
    }
    const bool skip_solver = nerr == 0;
    out.num_residuals = nerr;
    if (nerr == 0) {
      out.errs2.emplace_back(0);
      out.deltas2.emplace_back(0);
      out.successes.emplace_back(false);
      options.oss << TINYOPT_FORMAT("❌ #{}: No residuals, stopping", out.num_iters) << std::endl;
      // Can break only if first time, otherwise better count it as failure
      if (out.num_iters == 0) {
        out.stop_reason = OutputType::StopReason::kNoResiduals;
        break;
      }
    }

    // Damping
    for (int i = 0; i < size; ++i) JtJ(i, i) *= (1.0 + lambda);

    dX.setZero();
    bool solver_failed = skip_solver;
    bool system_has_nans = false;
    for (; !skip_solver && out.num_consec_failures <= max_tries;) {
      // Solver linear system
      if (options.ldlt) {
        const auto chol = Eigen::SelfAdjointView<const JtJ_t, Eigen::Upper>(JtJ).ldlt();
        if (chol.isPositive()) {
          dX = -chol.solve(Jt_res);
          solver_failed = false;
          break;
        } else {
          solver_failed = true;
          dX.setZero();
        }
      } else {  // Use Eigen's default inverse
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
          std::min(options.damping_range[1], std::max(options.damping_range[0], lambda * 10));
      const double s = (1.0 + lambda2) / (1.0 * lambda);
      options.oss << TINYOPT_FORMAT("❌ #{}: Cholesky Failed, redamping to λ:{:.2e}", out.num_iters,
                                    s)
                  << std::endl;
      for (int i = 0; i < size; ++i) JtJ(i, i) *= s;
      lambda = lambda2;
      out.num_consec_failures++;
      out.num_failures++;
    }

    // Check the displacement magnitude
    const double dX_norm2 = solver_failed ? 0 : dX.squaredNorm();
    const double Jt_res_norm2 = options.min_grad_norm2 == 0.0f ? 0 : Jt_res.squaredNorm();
    if (std::isnan(dX_norm2)) {
      solver_failed = true;
      options.oss << TINYOPT_FORMAT("❌ Failure, dX = \n{}", toString(dX.template cast<float>()))
                  << std::endl;
      options.oss << TINYOPT_FORMAT("JtJ = \n{}", toString(JtJ)) << std::endl;
      options.oss << TINYOPT_FORMAT("Jt*res = \n{}", toString(Jt_res)) << std::endl;
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
      if (out.num_iters > 0) X_last_good = X;
      // Move X by dX
      ptrait::pluseq(X, dX);
      // Save results
      out.last_err2 = err;
      if (options.export_JtJ) {
        out.last_JtJ = JtJ;  // TODO actually better store the one right after a success
        for (int i = 0; i < Size; ++i) out.last_JtJ(i, i) = JtJ(i, i) / (1.0f + lambda);
      }
      already_rolled_true = false;
      out.num_consec_failures = 0;
      if (options.log_x) {
        options.oss << TINYOPT_FORMAT(
                           "✅ #{}: X:{} |δX|:{:.2e} λ:{:.2e} ⎡σ⎤:{:.4f} "
                           "ε²:{:.5f} n:{} dε²:{:.3e} ∇ε²:{:.3e}",
                           out.num_iters, ptrait::toString(X), sqrt(dX_norm2), lambda,
                           sqrt(InvCov(JtJ).maxCoeff()), err, nerr, derr, Jt_res_norm2)
                    << std::endl;
      } else {
        options.oss << TINYOPT_FORMAT(
                           "✅ #{}: |δX|:{:.2e} λ:{:.2e} ε²:{:.5f} n:{} dε²:{:.3e} ∇ε²:{:.3e}",
                           out.num_iters, std::sqrt(dX_norm2), lambda, err, nerr, derr,
                           Jt_res_norm2)
                    << std::endl;
      }
      lambda = std::min(options.damping_range[1], std::max(options.damping_range[0], lambda / 3.0));
    } else { /* BAD Step */
      out.successes.emplace_back(false);
      if (options.log_x) {
        options.oss << TINYOPT_FORMAT(
                           "❌ #{}: X:{} |δX|:{:.2e} λ:{:.2e} ε²:{:.5f} n:{} dε²:{:.3e} ∇ε²:{:.3e}",
                           out.num_iters, ptrait::toString(X), sqrt(dX_norm2), lambda, err, nerr,
                           derr, Jt_res_norm2)
                    << std::endl;
      } else {
        options.oss << TINYOPT_FORMAT(
                           "❌ #{}: |δX|:{:.2e} λ:{:.2e} ε²:{:.5f} n:{} dε²:{:.3e} ∇ε²:{:.3e}",
                           out.num_iters, std::sqrt(dX_norm2), lambda, err, nerr, derr,
                           Jt_res_norm2)
                    << std::endl;
      }
      if (!already_rolled_true) {
        X = X_last_good;  // roll back by copy
        already_rolled_true = true;
      }
      out.num_failures++;
      out.num_consec_failures++;
      if (options.max_consec_failures > 0 &&
          out.num_consec_failures >= options.max_consec_failures) {
        out.stop_reason = OutputType::StopReason::kMaxConsecFails;
        break;
      }
      if (options.max_total_failures > 0 && out.num_failures >= options.max_total_failures) {
        out.stop_reason = OutputType::StopReason::kMaxFails;
        break;
      }
      lambda = std::min(options.damping_range[1], std::max(options.damping_range[0], lambda * 2));
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
 *  @brief Minimize a loss function @arg acc using the Levenberg-Marquardt minimization algorithm.
 *
 ***/
template <typename ParametersType, typename ResidualsFunc>
inline auto Optimize(ParametersType &x, ResidualsFunc &&func, const Options &options = Options{}) {
  if constexpr (std::is_invocable_v<ResidualsFunc, const ParametersType &>) {
    const auto optimize = [](auto &x, auto &&func, const auto &options) { return LM(x, func, options); };
    return tinyopt::OptimizeJet(x, func, optimize, options);
  } else {
    return LM(x, func, options);
  }
}

}  // namespace tinyopt::lm
