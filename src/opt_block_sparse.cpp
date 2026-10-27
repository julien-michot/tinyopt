// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <tinyopt/c/c_api_block_sparse_double.h>

#if TINYOPT_C_API_ENABLE_DYNAMIC

#if TINYOPT_C_API_ENABLE_FLOAT
#include <tinyopt/c/c_api_block_sparse_float.h>
#endif

#include <tinyopt/math/block_sparse_matrix.h>

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
tinyopt_status_t OptimizeBlockSparse(const Params *params, const Problem *problem,
                                     const tinyopt_options_t *c_options,
                                     tinyopt_summary_t *summary) {
  if (summary != nullptr) summary->final_covariance_dims = 0;
  if (params == nullptr || problem == nullptr || params->x == nullptr || params->dims <= 0 ||
      problem->acc_hessian == nullptr || problem->layout.block_sizes == nullptr ||
      problem->layout.block_counts == nullptr || problem->layout.num_block_types <= 0)
    return TINYOPT_STATUS_INVALID_ARGUMENT;

  std::vector<int> block_sizes(static_cast<std::size_t>(problem->layout.num_block_types));
  std::vector<Index> block_counts(static_cast<std::size_t>(problem->layout.num_block_types));
  Index dims = 0;
  for (int i = 0; i < problem->layout.num_block_types; ++i) {
    const int size = problem->layout.block_sizes[i];
    const int count = problem->layout.block_counts[i];
    if (size <= 0 || count < 0) return TINYOPT_STATUS_INVALID_ARGUMENT;
    block_sizes[static_cast<std::size_t>(i)] = size;
    block_counts[static_cast<std::size_t>(i)] = count;
    dims += static_cast<Index>(size) * count;
  }
  if (dims != params->dims) return TINYOPT_STATUS_INVALID_ARGUMENT;

  ParameterValues<Scalar> values;
  values.values.assign(params->x, params->x + params->dims);
  values.delta_values.resize(static_cast<std::size_t>(params->dims));
  values.plus_eq = params->plus_eq;

  try {
    Options options = ToTinyoptOptions(c_options);
    SparseMatrix<Scalar> empty(dims, dims);
    BlockSparseMatX<Scalar> prototype(std::move(empty), std::move(block_sizes),
                                      std::move(block_counts));
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
int SetBlock(Handle *hessian, int row, int col, const Scalar *data) {
  if (hessian == nullptr || row < 0 || col < 0) return 1;
  auto &matrix = *reinterpret_cast<BlockSparseMatX<Scalar> *>(hessian);
  try {
    matrix.setBlock(row, col, data);
    return 0;
  } catch (...) {
    return 1;
  }
}

}  // namespace

extern "C" int tinyopt_block_sparse_dims(const tinyopt_block_sparse_t *hessian) {
  return hessian != nullptr
             ? static_cast<int>(reinterpret_cast<const BlockSparseMatX<double> *>(hessian)->rows())
             : 0;
}

extern "C" int tinyopt_block_sparse_num_block_types(const tinyopt_block_sparse_t *hessian) {
  return hessian != nullptr
             ? reinterpret_cast<const BlockSparseMatX<double> *>(hessian)->num_block_types()
             : 0;
}

extern "C" int tinyopt_block_sparse_block_size(const tinyopt_block_sparse_t *hessian, int type) {
  return hessian != nullptr
             ? reinterpret_cast<const BlockSparseMatX<double> *>(hessian)->block_size(type)
             : 0;
}

extern "C" int tinyopt_block_sparse_num_blocks(const tinyopt_block_sparse_t *hessian, int type) {
  return hessian != nullptr
             ? static_cast<int>(
                   reinterpret_cast<const BlockSparseMatX<double> *>(hessian)->num_blocks(type))
             : 0;
}

extern "C" int tinyopt_block_sparse_block_offset(const tinyopt_block_sparse_t *hessian, int type) {
  return hessian != nullptr
             ? static_cast<int>(
                   reinterpret_cast<const BlockSparseMatX<double> *>(hessian)->base_offset(type))
             : 0;
}

extern "C" int tinyopt_block_sparse_block_dims(const tinyopt_block_sparse_t *hessian, int row,
                                               int col, int *rows, int *cols) {
  if (hessian == nullptr || rows == nullptr || cols == nullptr || row < 0 || col < 0) return 1;
  Index block_rows = 0;
  Index block_cols = 0;
  if (!reinterpret_cast<const BlockSparseMatX<double> *>(hessian)->block_dims(row, col, block_rows,
                                                                              block_cols))
    return 1;
  *rows = static_cast<int>(block_rows);
  *cols = static_cast<int>(block_cols);
  return 0;
}

extern "C" int tinyopt_block_sparse_set_block(tinyopt_block_sparse_t *hessian, int row, int col,
                                              const double *data) {
  return SetBlock<double>(hessian, row, col, data);
}

extern "C" TINYOPT_C_API tinyopt_status_t tinyopt_optimize_block_sparse(
    const tinyopt_params_t *params, const tinyopt_block_sparse_problem_t *problem,
    const tinyopt_options_t *options, tinyopt_summary_t *summary) {
  return OptimizeBlockSparse<double, tinyopt_block_sparse_t *>(params, problem, options, summary);
}

#if TINYOPT_C_API_ENABLE_FLOAT
extern "C" int tinyopt_block_sparsef_dims(const tinyopt_block_sparsef_t *hessian) {
  return hessian != nullptr
             ? static_cast<int>(reinterpret_cast<const BlockSparseMatX<float> *>(hessian)->rows())
             : 0;
}

extern "C" int tinyopt_block_sparsef_num_block_types(const tinyopt_block_sparsef_t *hessian) {
  return hessian != nullptr
             ? reinterpret_cast<const BlockSparseMatX<float> *>(hessian)->num_block_types()
             : 0;
}

extern "C" int tinyopt_block_sparsef_block_size(const tinyopt_block_sparsef_t *hessian, int type) {
  return hessian != nullptr
             ? reinterpret_cast<const BlockSparseMatX<float> *>(hessian)->block_size(type)
             : 0;
}

extern "C" int tinyopt_block_sparsef_num_blocks(const tinyopt_block_sparsef_t *hessian, int type) {
  return hessian != nullptr
             ? static_cast<int>(
                   reinterpret_cast<const BlockSparseMatX<float> *>(hessian)->num_blocks(type))
             : 0;
}

extern "C" int tinyopt_block_sparsef_block_offset(const tinyopt_block_sparsef_t *hessian,
                                                  int type) {
  return hessian != nullptr
             ? static_cast<int>(
                   reinterpret_cast<const BlockSparseMatX<float> *>(hessian)->base_offset(type))
             : 0;
}

extern "C" int tinyopt_block_sparsef_block_dims(const tinyopt_block_sparsef_t *hessian, int row,
                                                int col, int *rows, int *cols) {
  if (hessian == nullptr || rows == nullptr || cols == nullptr || row < 0 || col < 0) return 1;
  Index block_rows = 0;
  Index block_cols = 0;
  if (!reinterpret_cast<const BlockSparseMatX<float> *>(hessian)->block_dims(row, col, block_rows,
                                                                             block_cols))
    return 1;
  *rows = static_cast<int>(block_rows);
  *cols = static_cast<int>(block_cols);
  return 0;
}

extern "C" int tinyopt_block_sparsef_set_block(tinyopt_block_sparsef_t *hessian, int row, int col,
                                               const float *data) {
  return SetBlock<float>(hessian, row, col, data);
}

extern "C" TINYOPT_C_API tinyopt_status_t tinyopt_optimize_block_sparsef(
    const tinyopt_paramsf_t *params, const tinyopt_block_sparse_problemf_t *problem,
    const tinyopt_options_t *options, tinyopt_summary_t *summary) {
  return OptimizeBlockSparse<float, tinyopt_block_sparsef_t *>(params, problem, options, summary);
}
#endif /* TINYOPT_C_API_ENABLE_FLOAT */

#endif /* TINYOPT_C_API_ENABLE_DYNAMIC */
