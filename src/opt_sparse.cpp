// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <tinyopt/c/c_api_sparse.h>
#include <tinyopt/stop_reasons.h>

#include <algorithm>
#include <exception>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <tinyopt/optimize.h>

#include "c_api_options.h"
#include "c_api_params.h"
#include "c_api_problem.h"

namespace {

using tinyopt::c_api_detail::ParameterValues;
using tinyopt::c_api_detail::UserStopRequested;

struct CholmodCommon {
  cholmod_common value{};
  bool started = cholmod_start(&value) != 0;
  ~CholmodCommon() {
    if (started) cholmod_finish(&value);
  }
};

struct SparseDeleter {
  cholmod_common *common;
  void operator()(cholmod_sparse *matrix) const {
    if (matrix != nullptr) cholmod_free_sparse(&matrix, common);
  }
};

template <typename Scalar, typename Callback>
class SparseHessianAccumulator {
 public:
  using Triplet = Eigen::Triplet<Scalar>;

  SparseHessianAccumulator(Callback callback, void *user_data, cholmod_common *common)
      : callback_(callback), user_data_(user_data), common_(common) {}

  template <typename Params, typename Gradient, typename Hessian>
  Scalar operator()(const Params &params, Gradient &gradient, Hessian &hessian) const {
    Scalar cost = 0;
    constexpr bool kHasGradient = !tinyopt::traits::is_nullptr_v<Gradient>;
    constexpr bool kHasHessian = !tinyopt::traits::is_nullptr_v<Hessian>;
    Scalar *gradient_data = nullptr;
    cholmod_sparse *sparse = nullptr;
    cholmod_sparse **sparse_out = nullptr;
    if constexpr (kHasGradient) {
      gradient.setZero();
      gradient_data = gradient.data();
      if constexpr (kHasHessian) sparse_out = &sparse;
    }

    const int status = callback_(params.values.data(), static_cast<int>(params.dims()), &cost,
                                 gradient_data, sparse_out, common_, user_data_);
    std::unique_ptr<cholmod_sparse, SparseDeleter> owned(sparse, SparseDeleter{common_});
    if (status != 0) throw UserStopRequested{};
    if constexpr (kHasGradient && kHasHessian) {
      if (!owned) throw std::invalid_argument("Missing sparse Hessian");
      Convert(*owned, params.dims(), hessian);
    }
    return cost;
  }

 private:
  template <typename Hessian>
  void Convert(const cholmod_sparse &matrix, tinyopt::Index dims, Hessian &hessian) const {
    constexpr int kDtype = std::is_same_v<Scalar, double> ? CHOLMOD_DOUBLE : CHOLMOD_SINGLE;
    if (static_cast<tinyopt::Index>(matrix.nrow) != dims ||
        static_cast<tinyopt::Index>(matrix.ncol) != dims || matrix.itype != CHOLMOD_INT ||
        matrix.xtype != CHOLMOD_REAL || matrix.dtype != kDtype || matrix.p == nullptr ||
        matrix.i == nullptr || matrix.x == nullptr)
      throw std::invalid_argument("Invalid sparse Hessian");

    const int *p = static_cast<const int *>(matrix.p);
    const int *row_indices = static_cast<const int *>(matrix.i);
    const int *nz = static_cast<const int *>(matrix.nz);
    const Scalar *values = static_cast<const Scalar *>(matrix.x);
    triplets_.clear();
    for (int col = 0; col < static_cast<int>(matrix.ncol); ++col) {
      const int end = matrix.packed ? p[col + 1] : p[col] + nz[col];
      for (int k = p[col]; k < end; ++k) {
        const int row = row_indices[k];
        if (row < 0 || row >= static_cast<int>(matrix.nrow))
          throw std::invalid_argument("Invalid sparse Hessian index");
        // Keep a single triangle, stored as upper, as expected by the CHOLMOD linear solver
        if (matrix.stype < 0) {
          if (row >= col) triplets_.emplace_back(col, row, values[k]);
        } else if (row <= col) {
          triplets_.emplace_back(row, col, values[k]);
        }
      }
    }
    hessian.setFromTriplets(triplets_.begin(), triplets_.end());
  }

  Callback callback_;
  void *user_data_;
  cholmod_common *common_;
  mutable std::vector<Triplet> triplets_;
};

template <typename Scalar, typename Params, typename Problem>
tinyopt_status_t OptimizeSparse(const Params *params, const Problem *problem,
                                const tinyopt_options_t *c_options, tinyopt_summary_t *summary) {
  if (params == nullptr || problem == nullptr || params->x == nullptr || params->dims <= 0 ||
      problem->acc_hessian == nullptr)
    return TINYOPT_STATUS_INVALID_ARGUMENT;

  ParameterValues<Scalar> values;
  values.values.assign(params->x, params->x + params->dims);
  values.delta_values.resize(static_cast<std::size_t>(params->dims));
  values.plus_eq = params->plus_eq;

  CholmodCommon common;
  if (!common.started) return TINYOPT_STATUS_INTERNAL_ERROR;
  common.value.print = 0;

  try {
    tinyopt::Options options = tinyopt::c_api_detail::ToTinyoptOptions(c_options);
    options.linear_solver = tinyopt::LinearSolverMethod::SuiteSparse;
    SparseHessianAccumulator<Scalar, decltype(problem->acc_hessian)> accumulator(
        problem->acc_hessian, problem->user_data, &common.value);

    tinyopt::Summary result;
    switch (options.solver_type) {
      case tinyopt::Options::Solver::LevenbergMarquardt:
        result =
            tinyopt::lm::Optimizer<tinyopt::SparseMatrix<Scalar>>(options)(values, accumulator);
        break;
      case tinyopt::Options::Solver::GaussNewton:
        result =
            tinyopt::gn::Optimizer<tinyopt::SparseMatrix<Scalar>>(options)(values, accumulator);
        break;
      default:
        return TINYOPT_STATUS_INVALID_ARGUMENT;
    }

    if (summary != nullptr) {
      summary->stop_reason = static_cast<int>(result.stop_reason);
      summary->num_iters = result.num_iters;
      summary->num_failures = result.num_failures;
      summary->num_residuals = 1;
      summary->final_cost = result.final_cost;
      summary->used_numerical_differentiation = 0;
    }

    if (!result.Succeeded()) return TINYOPT_STATUS_OPTIMIZATION_FAILED;
    std::copy(values.values.begin(), values.values.end(), params->x);
    return TINYOPT_STATUS_OK;
  } catch (const UserStopRequested &) {
    std::copy(values.values.begin(), values.values.end(), params->x);
    if (summary != nullptr) {
      summary->stop_reason = static_cast<int>(tinyopt::StopReason::kUserStopped);
      summary->num_residuals = 1;
    }
    return TINYOPT_STATUS_USER_STOPPED;
  } catch (const std::invalid_argument &) {
    return TINYOPT_STATUS_INVALID_ARGUMENT;
  } catch (const std::exception &) {
    return TINYOPT_STATUS_INTERNAL_ERROR;
  } catch (...) {
    return TINYOPT_STATUS_INTERNAL_ERROR;
  }
}

}  // namespace

extern "C" TINYOPT_C_API tinyopt_status_t
tinyopt_optimize_sparse(const tinyopt_params_t *params, const tinyopt_sparse_problem_t *problem,
                        const tinyopt_options_t *options, tinyopt_summary_t *summary) {
  return OptimizeSparse<double>(params, problem, options, summary);
}

#if TINYOPT_C_API_ENABLE_FLOAT
extern "C" TINYOPT_C_API tinyopt_status_t
tinyopt_optimize_sparsef(const tinyopt_paramsf_t *params, const tinyopt_sparse_problemf_t *problem,
                         const tinyopt_options_t *options, tinyopt_summary_t *summary) {
  return OptimizeSparse<float>(params, problem, options, summary);
}
#endif
