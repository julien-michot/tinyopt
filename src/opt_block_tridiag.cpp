// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <tinyopt/c/c_api_block_tridiag_double.h>

#if TINYOPT_C_API_ENABLE_DYNAMIC

#if TINYOPT_C_API_ENABLE_FLOAT
#include <tinyopt/c/c_api_block_tridiag_float.h>
#endif

#include <tinyopt/math/block_tridiag_matrix.h>

#include "opt_block_common.h"

namespace {

using namespace tinyopt;
using tinyopt::c_api_detail::BlockHessianAccumulator;
using tinyopt::c_api_detail::CopyFinalCovariance;
using tinyopt::c_api_detail::OptimizeWithHessian;
using tinyopt::c_api_detail::ParameterValues;
using tinyopt::c_api_detail::ToTinyoptOptions;
using tinyopt::c_api_detail::UserStopRequested;

template <typename Scalar, typename Handle, typename Params, typename Problem>
tinyopt_status_t OptimizeBlockTridiag(const Params *params, const Problem *problem,
                                      const tinyopt_options_t *c_options,
                                      tinyopt_summary_t *summary) {
  if (summary != nullptr) summary->final_covariance_dims = 0;
  if (params == nullptr || problem == nullptr || params->x == nullptr || params->dims <= 0 ||
      problem->acc_hessian == nullptr || problem->num_blocks <= 0 || problem->block_size <= 0 ||
      static_cast<long long>(problem->num_blocks) * problem->block_size !=
          static_cast<long long>(params->dims))
    return TINYOPT_STATUS_INVALID_ARGUMENT;

  ParameterValues<Scalar> values;
  values.values.assign(params->x, params->x + params->dims);
  values.delta_values.resize(static_cast<std::size_t>(params->dims));
  values.plus_eq = params->plus_eq;

  try {
    Options options = ToTinyoptOptions(c_options);
    BlockTridiagMatX<Scalar> prototype(problem->num_blocks, problem->block_size);
    BlockHessianAccumulator<Scalar, Handle, decltype(problem->acc_hessian)> accumulator{
        problem->acc_hessian, problem->user_data};
    const auto result = OptimizeWithHessian(values, accumulator, options, std::move(prototype));

    if (summary != nullptr) {
      summary->stop_reason = static_cast<int>(result.stop_reason);
      summary->num_iters = result.num_iters;
      summary->num_failures = result.num_failures;
      summary->num_residuals = 1;
      summary->final_cost = result.final_cost;
      summary->used_numerical_differentiation = result.num_diff_used ? 1 : 0;
    }

    if (!result.Succeeded()) return TINYOPT_STATUS_OPTIMIZATION_FAILED;
    std::copy(values.values.begin(), values.values.end(), params->x);
    if (CopyFinalCovariance(result, c_options, summary)) return TINYOPT_STATUS_INSUFFICIENT_BUFFER;
    return TINYOPT_STATUS_OK;
  } catch (const UserStopRequested &) {
    std::copy(values.values.begin(), values.values.end(), params->x);
    if (summary != nullptr) {
      summary->stop_reason = static_cast<int>(StopReason::kUserStopped);
      summary->num_residuals = 1;
      summary->final_covariance_dims = 0;
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

template <typename Scalar, typename Handle>
int SetTridiagBlock(Handle *hessian, int row, int col, const Scalar *data) {
  if (hessian == nullptr || row < 0 || col < 0) return 1;
  auto &matrix = *reinterpret_cast<BlockTridiagMatX<Scalar> *>(hessian);
  if (row >= matrix.num_blocks() || col >= matrix.num_blocks()) return 1;
  const Index block_size = matrix.block_size();
  using BlockMap = Eigen::Map<const Matrix<Scalar, Dynamic, Dynamic>>;
  if (row == col) {
    if (data != nullptr) {
      matrix.diagonal(row) = BlockMap(data, block_size, block_size);
    } else {
      matrix.diagonal(row).setZero();
    }
    return 0;
  }
  if (col == row + 1) {
    if (data != nullptr) {
      matrix.off_diagonal(row) = BlockMap(data, block_size, block_size);
    } else {
      matrix.off_diagonal(row).setZero();
    }
    return 0;
  }
  if (row == col + 1) {
    if (data != nullptr) {
      matrix.off_diagonal(col) = BlockMap(data, block_size, block_size).transpose();
    } else {
      matrix.off_diagonal(col).setZero();
    }
    return 0;
  }
  return 1;
}

}  // namespace

extern "C" int tinyopt_block_tridiag_dims(const tinyopt_block_tridiag_t *hessian) {
  return hessian != nullptr
             ? static_cast<int>(reinterpret_cast<const BlockTridiagMatX<double> *>(hessian)->rows())
             : 0;
}

extern "C" int tinyopt_block_tridiag_block_size(const tinyopt_block_tridiag_t *hessian) {
  return hessian != nullptr
             ? static_cast<int>(
                   reinterpret_cast<const BlockTridiagMatX<double> *>(hessian)->block_size())
             : 0;
}

extern "C" int tinyopt_block_tridiag_num_blocks(const tinyopt_block_tridiag_t *hessian) {
  return hessian != nullptr
             ? static_cast<int>(
                   reinterpret_cast<const BlockTridiagMatX<double> *>(hessian)->num_blocks())
             : 0;
}

extern "C" int tinyopt_block_tridiag_set_block(tinyopt_block_tridiag_t *hessian, int row, int col,
                                               const double *data) {
  return SetTridiagBlock<double>(hessian, row, col, data);
}

extern "C" TINYOPT_C_API tinyopt_status_t tinyopt_optimize_block_tridiag(
    const tinyopt_params_t *params, const tinyopt_block_tridiag_problem_t *problem,
    const tinyopt_options_t *options, tinyopt_summary_t *summary) {
  return OptimizeBlockTridiag<double, tinyopt_block_tridiag_t *>(params, problem, options, summary);
}

#if TINYOPT_C_API_ENABLE_FLOAT
extern "C" int tinyopt_block_tridiagf_dims(const tinyopt_block_tridiagf_t *hessian) {
  return hessian != nullptr
             ? static_cast<int>(reinterpret_cast<const BlockTridiagMatX<float> *>(hessian)->rows())
             : 0;
}

extern "C" int tinyopt_block_tridiagf_block_size(const tinyopt_block_tridiagf_t *hessian) {
  return hessian != nullptr
             ? static_cast<int>(
                   reinterpret_cast<const BlockTridiagMatX<float> *>(hessian)->block_size())
             : 0;
}

extern "C" int tinyopt_block_tridiagf_num_blocks(const tinyopt_block_tridiagf_t *hessian) {
  return hessian != nullptr
             ? static_cast<int>(
                   reinterpret_cast<const BlockTridiagMatX<float> *>(hessian)->num_blocks())
             : 0;
}

extern "C" int tinyopt_block_tridiagf_set_block(tinyopt_block_tridiagf_t *hessian, int row, int col,
                                                const float *data) {
  return SetTridiagBlock<float>(hessian, row, col, data);
}

extern "C" TINYOPT_C_API tinyopt_status_t tinyopt_optimize_block_tridiagf(
    const tinyopt_paramsf_t *params, const tinyopt_block_tridiag_problemf_t *problem,
    const tinyopt_options_t *options, tinyopt_summary_t *summary) {
  return OptimizeBlockTridiag<float, tinyopt_block_tridiagf_t *>(params, problem, options, summary);
}
#endif /* TINYOPT_C_API_ENABLE_FLOAT */

#endif /* TINYOPT_C_API_ENABLE_DYNAMIC */
