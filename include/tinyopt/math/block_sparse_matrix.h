// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <tinyopt/traits/matrix_traits.h>
#include <tinyopt/types.h>

namespace tinyopt {

/// Stores the coordinates (row, col) of an allocated block.
struct BlockIndex {
  Index row = 0;
  Index col = 0;

  constexpr BlockIndex() = default;
  constexpr BlockIndex(Index r, Index c) : row(r), col(c) {}
  constexpr BlockIndex(const std::pair<Index, Index>& p) : row(p.first), col(p.second) {}

  constexpr operator std::pair<Index, Index>() const { return {row, col}; }

  [[nodiscard]] constexpr Index first() const noexcept { return row; }
  [[nodiscard]] constexpr Index second() const noexcept { return col; }

  constexpr bool operator==(const BlockIndex& other) const noexcept = default;
};

template <typename Scalar, int... BlockSizes>
class BlockSparseMatrix;

/// Runtime-layout block-sparse matrix. Block sizes and counts are selected at runtime, which makes
/// it suitable for the C API and for storing block-structured covariances of heterogeneous
/// problems. The scalar matrix is the canonical storage; the block layout maps global block
/// indices to scalar offsets.
template <typename Scalar>
class BlockSparseMatrix<Scalar, Dynamic> {
 public:
  using ScalarType = Scalar;
  using SparseMatrixType = SparseMatrix<Scalar>;

  BlockSparseMatrix() = default;

  BlockSparseMatrix(SparseMatrixType matrix, std::vector<int> block_sizes,
                    std::vector<Index> block_counts)
      : matrix_(std::move(matrix)),
        block_sizes_(std::move(block_sizes)),
        block_counts_(std::move(block_counts)) {
    ValidateLayout();
  }

  [[nodiscard]] Index rows() const noexcept { return matrix_.rows(); }
  [[nodiscard]] Index cols() const noexcept { return matrix_.cols(); }
  [[nodiscard]] Index size() const noexcept { return matrix_.size(); }
  [[nodiscard]] Index dims() const noexcept { return matrix_.rows(); }
  [[nodiscard]] const std::vector<int>& block_sizes() const noexcept { return block_sizes_; }
  [[nodiscard]] const std::vector<Index>& block_counts() const noexcept { return block_counts_; }
  [[nodiscard]] int num_block_types() const noexcept {
    return static_cast<int>(block_sizes_.size());
  }

  [[nodiscard]] int block_size(int type) const noexcept {
    return type >= 0 && type < num_block_types() ? block_sizes_[static_cast<size_t>(type)] : 0;
  }

  [[nodiscard]] Index num_blocks(int type) const noexcept {
    return type >= 0 && type < num_block_types() ? block_counts_[static_cast<size_t>(type)]
                                                 : Index(0);
  }

  /// Scalar offset of the first block of the given type.
  [[nodiscard]] Index base_offset(int type) const noexcept {
    Index offset = 0;
    for (int i = 0; i < type && i < num_block_types(); ++i)
      offset += static_cast<Index>(block_sizes_[static_cast<size_t>(i)]) *
                block_counts_[static_cast<size_t>(i)];
    return offset;
  }

  /// Block offset of the first block of the given type, in the global block index space.
  [[nodiscard]] Index base_block_offset(int type) const noexcept {
    Index offset = 0;
    for (int i = 0; i < type && i < num_block_types(); ++i)
      offset += block_counts_[static_cast<size_t>(i)];
    return offset;
  }

  /// Number of blocks across all block types.
  [[nodiscard]] Index num_blocks() const noexcept {
    Index total = 0;
    for (const Index count : block_counts_) total += count;
    return total;
  }

  [[nodiscard]] const SparseMatrixType& toSparse() const noexcept { return matrix_; }
  [[nodiscard]] SparseMatrixType& toSparse() noexcept { return matrix_; }
  [[nodiscard]] Matrix<Scalar, Dynamic, Dynamic> toDense() const {
    Matrix<Scalar, Dynamic, Dynamic> dense = Matrix<Scalar, Dynamic, Dynamic>(matrix_);
    for (int outer = 0; outer < matrix_.outerSize(); ++outer) {
      for (typename SparseMatrixType::InnerIterator it(matrix_, outer); it; ++it) {
        dense(it.col(), it.row()) = it.value();
      }
    }
    return dense;
  }
  [[nodiscard]] BlockSparseMatrix toBlockSparseMatX() const { return *this; }
  [[nodiscard]] BlockSparseMatrix transpose() const { return *this; }
  [[nodiscard]] Index nonZeros() const { return matrix_.nonZeros(); }
  void makeCompressed() { matrix_.makeCompressed(); }

  /// Resolves a global block index to its block type and local block index.
  [[nodiscard]] bool BlockCoords(Index global_block, int& type, Index& local) const noexcept {
    Index base = 0;
    for (size_t t = 0; t < block_counts_.size(); ++t) {
      if (global_block < base + block_counts_[t]) {
        type = static_cast<int>(t);
        local = global_block - base;
        return true;
      }
      base += block_counts_[t];
    }
    return false;
  }

  /// Returns the dimensions of the block at global block coordinates (row, col).
  [[nodiscard]] bool block_dims(Index row, Index col, Index& block_rows,
                                Index& block_cols) const noexcept {
    int row_type = 0;
    int col_type = 0;
    Index row_local = 0;
    Index col_local = 0;
    if (!BlockCoords(row, row_type, row_local) || !BlockCoords(col, col_type, col_local))
      return false;
    block_rows = block_sizes_[static_cast<size_t>(row_type)];
    block_cols = block_sizes_[static_cast<size_t>(col_type)];
    return true;
  }

  /// Writes a column-major block at global block coordinates (row, col). Passing NULL zeroes the
  /// block. Lower-triangular coordinates are mirrored into the stored upper triangle.
  void setBlock(Index row, Index col, const Scalar* data) {
    int row_type = 0;
    int col_type = 0;
    Index row_local = 0;
    Index col_local = 0;
    if (!BlockCoords(row, row_type, row_local) || !BlockCoords(col, col_type, col_local))
      throw std::out_of_range("Block-sparse block index out of range");
    const Index block_rows = block_sizes_[static_cast<size_t>(row_type)];
    const Index block_cols = block_sizes_[static_cast<size_t>(col_type)];
    const Index row_offset = base_offset(row_type) + row_local * block_rows;
    const Index col_offset = base_offset(col_type) + col_local * block_cols;
    if (row <= col) {
      for (Index i = 0; i < block_rows; ++i) {
        for (Index j = 0; j < block_cols; ++j) {
          const Scalar value = data != nullptr ? data[i + j * block_rows] : Scalar(0);
          matrix_.coeffRef(row_offset + i, col_offset + j) = value;
        }
      }
    } else {
      for (Index i = 0; i < block_rows; ++i) {
        for (Index j = 0; j < block_cols; ++j) {
          const Scalar value = data != nullptr ? data[i + j * block_rows] : Scalar(0);
          matrix_.coeffRef(col_offset + j, row_offset + i) = value;
        }
      }
    }
    if (!matrix_.isCompressed()) matrix_.makeCompressed();
  }

  /// Optimizer interface: clears values while preserving the block layout.
  void clear() noexcept {
    matrix_.setZero();
    matrix_.makeCompressed();
  }

  void setZero() noexcept { matrix_.setZero(); }

  [[nodiscard]] Vector<Scalar, Dynamic> diagonal() const {
    Vector<Scalar, Dynamic> result = Vector<Scalar, Dynamic>::Zero(rows());
    for (Index i = 0; i < rows(); ++i) result[i] = matrix_.coeff(i, i);
    return result;
  }

  void scaleDiagonal(Scalar factor) {
    for (Index i = 0; i < rows(); ++i) matrix_.coeffRef(i, i) *= factor;
  }

  void scaleByDiagonal(const Vector<Scalar, Dynamic>& scale, bool inverse = false) {
    for (int k = 0; k < matrix_.outerSize(); ++k) {
      for (typename SparseMatrixType::InnerIterator it(matrix_, k); it; ++it) {
        const Scalar factor = scale[it.row()] * scale[it.col()];
        it.valueRef() = inverse ? it.value() / factor : it.value() * factor;
      }
    }
  }

  void applyFixedIndices(const std::vector<Index>& fixed_indices) {
    if (fixed_indices.empty()) return;
    std::vector<bool> fixed(static_cast<size_t>(rows()), false);
    for (const Index index : fixed_indices) {
      if (index >= 0 && index < rows()) fixed[static_cast<size_t>(index)] = true;
    }
    for (int k = 0; k < matrix_.outerSize(); ++k) {
      for (typename SparseMatrixType::InnerIterator it(matrix_, k); it; ++it) {
        if (fixed[static_cast<size_t>(it.row())] || fixed[static_cast<size_t>(it.col())])
          it.valueRef() = it.row() == it.col() ? Scalar(1) : Scalar(0);
      }
    }
    for (const Index index : fixed_indices) {
      if (index >= 0 && index < rows()) matrix_.coeffRef(index, index) = Scalar(1);
    }
  }

  [[nodiscard]] Vector<Scalar, Dynamic> operator*(const Vector<Scalar, Dynamic>& vector) const {
    return matrix_.template selfadjointView<Eigen::Upper>() * vector;
  }

  [[nodiscard]] Scalar squaredNorm() const noexcept { return matrix_.squaredNorm(); }
  [[nodiscard]] Scalar norm() const noexcept { return std::sqrt(squaredNorm()); }

  template <typename OtherScalar>
  [[nodiscard]] BlockSparseMatrix<OtherScalar, Dynamic> cast() const {
    return BlockSparseMatrix<OtherScalar, Dynamic>(matrix_.template cast<OtherScalar>(),
                                                   block_sizes_, block_counts_);
  }

 private:
  void ValidateLayout() {
    if (block_sizes_.empty() || block_sizes_.size() != block_counts_.size()) {
      throw std::invalid_argument("Block-sparse layout must have matching non-empty block types");
    }
    Index dims = 0;
    for (size_t i = 0; i < block_sizes_.size(); ++i) {
      if (block_sizes_[i] <= 0 || block_counts_[i] < 0) {
        throw std::invalid_argument(
            "Block-sparse block sizes must be positive and block counts non-negative");
      }
      dims += static_cast<Index>(block_sizes_[i]) * block_counts_[i];
    }
    if (matrix_.rows() != dims || matrix_.cols() != dims) {
      throw std::invalid_argument("Block-sparse layout dimensions must match the matrix");
    }
  }

  SparseMatrixType matrix_;
  std::vector<int> block_sizes_;
  std::vector<Index> block_counts_;
};

template <typename Scalar = double>
using BlockSparseMatX = BlockSparseMatrix<Scalar, Dynamic>;

namespace internal {

// Pair index mapping for upper-triangular block types (0 <= i <= j < N)
constexpr size_t PairIndex(size_t i, size_t j, size_t N) noexcept {
  return i * N - (i * (i - 1)) / 2 + (j - i);
}

// Convert flat pair index K to (i, j) with i <= j
template <size_t N>
constexpr std::pair<size_t, size_t> KToPair(size_t k) noexcept {
  size_t count = 0;
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = i; j < N; ++j) {
      if (count == k) return {i, j};
      ++count;
    }
  }
  return {0, 0};
}

/// Storage for a single block type pair (Rows x Cols)
template <typename Scalar, int Rows, int Cols>
struct BlockPairStorage {
  using MatrixType = Matrix<Scalar, Rows, Cols>;

  std::vector<MatrixType> blocks;
  std::vector<BlockIndex> coords;

  // Direct 1D lookup table for diagonal blocks when Rows == Cols
  std::vector<int> diag_grid;

  // Direct 2D lookup table: index_grid[r * grid_cols + c] gives index in blocks, or -1.
  std::vector<int> index_grid;
  Index grid_rows = 0;
  Index grid_cols = 0;

  // Sparse hash table fallback when dense grid would exceed max allowed elements
  std::unordered_map<uint64_t, int> sparse_grid;
  static constexpr uint64_t kMaxDenseGridElements = 16'000'000;

  void EnsureGrid(Index r, Index c) {
    if (r >= grid_rows || c >= grid_cols) {
      Index new_rows =
          (r >= grid_rows) ? std::max(r + 1, grid_rows == 0 ? Index(4) : grid_rows * 2) : grid_rows;
      Index new_cols =
          (c >= grid_cols) ? std::max(c + 1, grid_cols == 0 ? Index(4) : grid_cols * 2) : grid_cols;
      std::vector<int> new_grid(static_cast<size_t>(new_rows * new_cols), -1);
      for (size_t k = 0; k < coords.size(); ++k) {
        if constexpr (Rows == Cols) {
          if (coords[k].row == coords[k].col) continue;
        }
        if (coords[k].row < new_rows && coords[k].col < new_cols) {
          new_grid[static_cast<size_t>(coords[k].row * new_cols + coords[k].col)] =
              static_cast<int>(k);
        }
      }
      index_grid = std::move(new_grid);
      grid_rows = new_rows;
      grid_cols = new_cols;
    }
  }

  [[nodiscard]] int GetIndex(Index r, Index c) const noexcept {
    if constexpr (Rows == Cols) {
      if (r == c) {
        if (static_cast<size_t>(r) < diag_grid.size()) {
          return diag_grid[static_cast<size_t>(r)];
        }
        return -1;
      }
    }
    if (r < grid_rows && c < grid_cols && !index_grid.empty()) {
      return index_grid[static_cast<size_t>(r * grid_cols + c)];
    }
    if (!sparse_grid.empty()) {
      const uint64_t key = (static_cast<uint64_t>(r) << 32) | static_cast<uint32_t>(c);
      auto it = sparse_grid.find(key);
      if (it != sparse_grid.end()) return it->second;
    }
    return -1;
  }

  [[nodiscard]] bool HasBlock(Index r, Index c) const noexcept { return GetIndex(r, c) != -1; }

  MatrixType& GetOrCreateBlock(Index r, Index c) {
    int idx = GetIndex(r, c);
    if (idx != -1) {
      return blocks[static_cast<size_t>(idx)];
    }
    int new_idx = static_cast<int>(blocks.size());
    blocks.emplace_back(MatrixType::Zero());
    coords.emplace_back(r, c);

    if constexpr (Rows == Cols) {
      if (r == c) {
        if (static_cast<size_t>(r) >= diag_grid.size()) {
          diag_grid.resize(
              static_cast<size_t>(std::max(
                  r + 1, diag_grid.empty() ? Index(4) : static_cast<Index>(diag_grid.size()) * 2)),
              -1);
        }
        diag_grid[static_cast<size_t>(r)] = new_idx;
        return blocks[static_cast<size_t>(new_idx)];
      }
    }

    Index new_rows =
        (r >= grid_rows) ? std::max(r + 1, grid_rows == 0 ? Index(4) : grid_rows * 2) : grid_rows;
    Index new_cols =
        (c >= grid_cols) ? std::max(c + 1, grid_cols == 0 ? Index(4) : grid_cols * 2) : grid_cols;
    if (static_cast<uint64_t>(new_rows) * static_cast<uint64_t>(new_cols) <=
        kMaxDenseGridElements) {
      EnsureGrid(r, c);
      index_grid[static_cast<size_t>(r * grid_cols + c)] = new_idx;
    } else {
      const uint64_t key = (static_cast<uint64_t>(r) << 32) | static_cast<uint32_t>(c);
      sparse_grid[key] = new_idx;
    }
    return blocks[static_cast<size_t>(new_idx)];
  }

  [[nodiscard]] const MatrixType& GetBlockOrZero(Index r, Index c) const noexcept {
    int idx = GetIndex(r, c);
    if (idx != -1) {
      return blocks[static_cast<size_t>(idx)];
    }
    static const MatrixType zero = MatrixType::Zero();
    return zero;
  }

  void Clear() noexcept {
    if (!index_grid.empty()) {
      for (const auto& coord : coords) {
        if constexpr (Rows == Cols) {
          if (coord.row == coord.col) continue;
        }
        if (coord.row < grid_rows && coord.col < grid_cols) {
          index_grid[static_cast<size_t>(coord.row * grid_cols + coord.col)] = -1;
        }
      }
    }
    if constexpr (Rows == Cols) {
      for (const auto& coord : coords) {
        if (coord.row == coord.col && static_cast<size_t>(coord.row) < diag_grid.size()) {
          diag_grid[static_cast<size_t>(coord.row)] = -1;
        }
      }
    }
    blocks.clear();
    coords.clear();
    sparse_grid.clear();
  }

  void SetZero() noexcept {
    for (auto& b : blocks) {
      b.setZero();
    }
  }
};

template <typename Scalar, int... Sizes>
struct StorageTupleHelper {
  static constexpr size_t N = sizeof...(Sizes);
  static constexpr int kSizes[N] = {Sizes...};
  static constexpr size_t kNumPairs = N * (N + 1) / 2;

  template <size_t K>
  static constexpr int GetFirstSize() {
    constexpr auto pr = KToPair<N>(K);
    return kSizes[pr.first];
  }

  template <size_t K>
  static constexpr int GetSecondSize() {
    constexpr auto pr = KToPair<N>(K);
    return kSizes[pr.second];
  }

  template <size_t... Ks>
  static auto MakeTupleType(std::index_sequence<Ks...>)
      -> std::tuple<BlockPairStorage<Scalar, GetFirstSize<Ks>(), GetSecondSize<Ks>()>...>;

  using Type = decltype(MakeTupleType(std::make_index_sequence<kNumPairs>{}));
};

}  // namespace internal

/**
 * @brief High-performance symmetric Block Sparse Matrix storing fixed-size blocks.
 *
 * Template parameters:
 *   - Scalar: scalar type (double, float, etc.)
 *   - BlockSizes: integer sizes S1, S2, ..., SN of the block types.
 *
 * For all pairs (Si, Sj) with i <= j, blocks of size Si x Sj are stored in contiguous
 * std::vector<Matrix<Scalar, Si, Sj>> (half of all permutations, upper triangular only).
 *
 * RW access via .block<S1, S2>(r, c) provides compile-time read/write access.
 * Lower-triangular access via .block<S2, S1>(c, r) returns an Eigen Transpose view
 * (an expression glue class with zero copy) on the stored .block<S1, S2>(r, c).
 */
template <typename Scalar, int... BlockSizes>
class BlockSparseMatrix {
 public:
  using ScalarType = Scalar;
  static constexpr size_t kNumBlockTypes = sizeof...(BlockSizes);
  static constexpr int kBlockSizes[kNumBlockTypes] = {BlockSizes...};
  static constexpr size_t kNumPairs = kNumBlockTypes * (kNumBlockTypes + 1) / 2;

  using StorageTuple = typename internal::StorageTupleHelper<Scalar, BlockSizes...>::Type;

  // ---------------------------------------------------------------------------
  // Constructors
  // ---------------------------------------------------------------------------

  BlockSparseMatrix() { block_counts_.fill(0); }

  template <typename... CountTypes>
    requires(sizeof...(CountTypes) == kNumBlockTypes &&
             (std::is_integral_v<std::decay_t<CountTypes>> && ...))
  explicit BlockSparseMatrix(CountTypes... counts) {
    Index arr[] = {static_cast<Index>(counts)...};
    for (size_t i = 0; i < kNumBlockTypes; ++i) {
      block_counts_[i] = arr[i];
    }
  }

  explicit BlockSparseMatrix(const std::array<Index, kNumBlockTypes>& counts)
      : block_counts_(counts) {}

  // ---------------------------------------------------------------------------
  // Dimensions and Sizing
  // ---------------------------------------------------------------------------

  [[nodiscard]] static constexpr size_t num_block_types() noexcept { return kNumBlockTypes; }

  template <int S>
  [[nodiscard]] static constexpr size_t find_type_index() noexcept {
    for (size_t i = 0; i < kNumBlockTypes; ++i) {
      if (kBlockSizes[i] == S) return i;
    }
    return 0;
  }

  [[nodiscard]] Index num_blocks(size_t type_idx) const noexcept {
    Index max_c = block_counts_[type_idx];
    auto check_storage = [&]<size_t K>(std::integral_constant<size_t, K>) {
      constexpr auto pr = internal::KToPair<kNumBlockTypes>(K);
      if (pr.first == type_idx || pr.second == type_idx) {
        const auto& st = std::get<K>(storages_);
        for (const auto& p : st.coords) {
          if (pr.first == type_idx) max_c = std::max(max_c, p.row + 1);
          if (pr.second == type_idx) max_c = std::max(max_c, p.col + 1);
        }
      }
    };
    [&]<size_t... Ks>(std::index_sequence<Ks...>) {
      (check_storage(std::integral_constant<size_t, Ks>{}), ...);
    }(std::make_index_sequence<kNumPairs>{});
    block_counts_[type_idx] = max_c;
    return max_c;
  }

  template <int S>
  [[nodiscard]] Index num_blocks() const noexcept {
    return num_blocks(find_type_index<S>());
  }

  [[nodiscard]] std::array<Index, kNumBlockTypes> block_counts() const noexcept {
    std::array<Index, kNumBlockTypes> counts{};
    for (size_t i = 0; i < kNumBlockTypes; ++i) {
      counts[i] = num_blocks(i);
    }
    return counts;
  }

  [[nodiscard]] Index block_rows() const noexcept {
    Index total = 0;
    for (size_t i = 0; i < kNumBlockTypes; ++i) {
      total += num_blocks(i);
    }
    return total;
  }

  [[nodiscard]] Index block_cols() const noexcept { return block_rows(); }

  [[nodiscard]] Index rows() const noexcept {
    Index total = 0;
    for (size_t i = 0; i < kNumBlockTypes; ++i) {
      total += num_blocks(i) * kBlockSizes[i];
    }
    return total;
  }

  [[nodiscard]] Index cols() const noexcept { return rows(); }

  [[nodiscard]] Index size() const noexcept { return rows() * cols(); }
  [[nodiscard]] Index dims() const noexcept { return rows(); }

  [[nodiscard]] Index base_offset(size_t type_idx) const noexcept {
    Index off = 0;
    for (size_t i = 0; i < type_idx; ++i) {
      off += num_blocks(i) * kBlockSizes[i];
    }
    return off;
  }

  template <typename... CountTypes>
    requires(sizeof...(CountTypes) == kNumBlockTypes &&
             (std::is_integral_v<std::decay_t<CountTypes>> && ...))
  void resize(CountTypes... counts) {
    Index arr[] = {static_cast<Index>(counts)...};
    for (size_t i = 0; i < kNumBlockTypes; ++i) {
      block_counts_[i] = arr[i];
    }
  }

  void resize(const std::array<Index, kNumBlockTypes>& counts) { block_counts_ = counts; }

  void clear() noexcept {
    block_counts_.fill(0);
    [&]<size_t... Ks>(std::index_sequence<Ks...>) {
      (std::get<Ks>(storages_).Clear(), ...);
    }(std::make_index_sequence<kNumPairs>{});
  }

  void setZero() noexcept {
    [&]<size_t... Ks>(std::index_sequence<Ks...>) {
      (std::get<Ks>(storages_).SetZero(), ...);
    }(std::make_index_sequence<kNumPairs>{});
  }

  void setIdentity() {
    setZero();
    auto set_id = [&]<size_t K>(std::integral_constant<size_t, K>) {
      constexpr auto pr = internal::KToPair<kNumBlockTypes>(K);
      if constexpr (pr.first == pr.second) {
        Index n = num_blocks(pr.first);
        for (Index i = 0; i < n; ++i) {
          block_by_type<pr.first, pr.first>(i, i).setIdentity();
        }
      }
    };
    [&]<size_t... Ks>(std::index_sequence<Ks...>) {
      (set_id(std::integral_constant<size_t, Ks>{}), ...);
    }(std::make_index_sequence<kNumPairs>{});
  }

  [[nodiscard]] size_t num_allocated_blocks() const noexcept {
    size_t count = 0;
    [&]<size_t... Ks>(std::index_sequence<Ks...>) {
      ((count += std::get<Ks>(storages_).blocks.size()), ...);
    }(std::make_index_sequence<kNumPairs>{});
    return count;
  }

  template <int Rows, int Cols>
  [[nodiscard]] size_t num_allocated_blocks() const noexcept {
    constexpr size_t type_r = find_type_index<Rows>();
    constexpr size_t type_c = find_type_index<Cols>();
    constexpr size_t ti = std::min(type_r, type_c);
    constexpr size_t tj = std::max(type_r, type_c);
    constexpr size_t P = internal::PairIndex(ti, tj, kNumBlockTypes);
    return std::get<P>(storages_).blocks.size();
  }

  [[nodiscard]] Index nonZeros() const noexcept {
    Index total = 0;
    auto count_nz = [&]<size_t K>(std::integral_constant<size_t, K>) {
      constexpr auto pr = internal::KToPair<kNumBlockTypes>(K);
      const auto& st = std::get<K>(storages_);
      if constexpr (pr.first == pr.second) {
        for (size_t k = 0; k < st.blocks.size(); ++k) {
          if (st.coords[k].row == st.coords[k].col) {
            total += (st.blocks[k].array() != Scalar(0)).count();
          } else {
            total += 2 * (st.blocks[k].array() != Scalar(0)).count();
          }
        }
      } else {
        for (size_t k = 0; k < st.blocks.size(); ++k) {
          total += 2 * (st.blocks[k].array() != Scalar(0)).count();
        }
      }
    };
    [&]<size_t... Ks>(std::index_sequence<Ks...>) {
      (count_nz(std::integral_constant<size_t, Ks>{}), ...);
    }(std::make_index_sequence<kNumPairs>{});
    return total;
  }

  [[nodiscard]] bool empty() const noexcept { return num_allocated_blocks() == 0; }

  // ---------------------------------------------------------------------------
  // Block Access Methods (Compile-time RW, Transposed View, Eigen-style)
  // ---------------------------------------------------------------------------

  template <size_t TypeRow, size_t TypeCol>
  decltype(auto) block_by_type(Index r, Index c) {
    block_counts_[TypeRow] = std::max(block_counts_[TypeRow], r + 1);
    block_counts_[TypeCol] = std::max(block_counts_[TypeCol], c + 1);
    constexpr int Rows = kBlockSizes[TypeRow];
    constexpr int Cols = kBlockSizes[TypeCol];

    if constexpr (TypeRow < TypeCol) {
      constexpr size_t P = internal::PairIndex(TypeRow, TypeCol, kNumBlockTypes);
      return (std::get<P>(storages_).GetOrCreateBlock(r, c));
    } else if constexpr (TypeRow > TypeCol) {
      constexpr size_t P = internal::PairIndex(TypeCol, TypeRow, kNumBlockTypes);
      return std::get<P>(storages_).GetOrCreateBlock(c, r).transpose();
    } else {
      constexpr size_t P = internal::PairIndex(TypeRow, TypeRow, kNumBlockTypes);
      if (r <= c) {
        auto blk_data = std::get<P>(storages_).GetOrCreateBlock(r, c).data();
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic> stride(Rows, 1);
        using MapType = Eigen::Map<Matrix<Scalar, Rows, Cols>, 0,
                                   Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;
        return MapType(blk_data, stride);
      } else {
        auto blk_data = std::get<P>(storages_).GetOrCreateBlock(c, r).data();
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic> stride(1, Rows);
        using MapType = Eigen::Map<Matrix<Scalar, Rows, Cols>, 0,
                                   Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;
        return MapType(blk_data, stride);
      }
    }
  }

  template <size_t TypeRow, size_t TypeCol>
  decltype(auto) block_by_type(Index r, Index c) const {
    constexpr int Rows = kBlockSizes[TypeRow];
    constexpr int Cols = kBlockSizes[TypeCol];

    if constexpr (TypeRow < TypeCol) {
      constexpr size_t P = internal::PairIndex(TypeRow, TypeCol, kNumBlockTypes);
      return (std::get<P>(storages_).GetBlockOrZero(r, c));
    } else if constexpr (TypeRow > TypeCol) {
      constexpr size_t P = internal::PairIndex(TypeCol, TypeRow, kNumBlockTypes);
      return std::get<P>(storages_).GetBlockOrZero(c, r).transpose();
    } else {
      constexpr size_t P = internal::PairIndex(TypeRow, TypeRow, kNumBlockTypes);
      if (r <= c) {
        const auto blk_data = std::get<P>(storages_).GetBlockOrZero(r, c).data();
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic> stride(Rows, 1);
        using MapType = Eigen::Map<const Matrix<Scalar, Rows, Cols>, 0,
                                   Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;
        return MapType(blk_data, stride);
      } else {
        const auto blk_data = std::get<P>(storages_).GetBlockOrZero(c, r).data();
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic> stride(1, Rows);
        using MapType = Eigen::Map<const Matrix<Scalar, Rows, Cols>, 0,
                                   Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;
        return MapType(blk_data, stride);
      }
    }
  }

  template <int Rows, int Cols>
  decltype(auto) block(Index r, Index c) {
    constexpr size_t type_r = find_type_index<Rows>();
    constexpr size_t type_c = find_type_index<Cols>();
    return block_by_type<type_r, type_c>(r, c);
  }

  template <int Rows, int Cols>
  decltype(auto) block(Index r, Index c) const {
    constexpr size_t type_r = find_type_index<Rows>();
    constexpr size_t type_c = find_type_index<Cols>();
    return block_by_type<type_r, type_c>(r, c);
  }

  decltype(auto) block(Index r, Index c)
    requires(kNumBlockTypes == 1)
  {
    return block<kBlockSizes[0], kBlockSizes[0]>(r, c);
  }

  decltype(auto) block(Index r, Index c) const
    requires(kNumBlockTypes == 1)
  {
    return block<kBlockSizes[0], kBlockSizes[0]>(r, c);
  }

  template <int S>
  Matrix<Scalar, S, S>& diagonal(Index i) {
    constexpr size_t type_idx = find_type_index<S>();
    block_counts_[type_idx] = std::max(block_counts_[type_idx], i + 1);
    constexpr size_t P = internal::PairIndex(type_idx, type_idx, kNumBlockTypes);
    return std::get<P>(storages_).GetOrCreateBlock(i, i);
  }

  template <int S>
  const Matrix<Scalar, S, S>& diagonal(Index i) const {
    constexpr size_t type_idx = find_type_index<S>();
    constexpr size_t P = internal::PairIndex(type_idx, type_idx, kNumBlockTypes);
    return std::get<P>(storages_).GetBlockOrZero(i, i);
  }

  template <int Rows, int Cols>
  decltype(auto) topLeftCorner() {
    return block<Rows, Cols>(0, 0);
  }

  template <int Rows, int Cols>
  decltype(auto) topLeftCorner() const {
    return block<Rows, Cols>(0, 0);
  }

  template <int Rows, int Cols>
  decltype(auto) topRightCorner(Index c) {
    return block<Rows, Cols>(0, c);
  }

  template <int Rows, int Cols>
  decltype(auto) topRightCorner(Index c) const {
    return block<Rows, Cols>(0, c);
  }

  template <int Rows, int Cols>
  decltype(auto) bottomLeftCorner(Index r) {
    return block<Rows, Cols>(r, 0);
  }

  template <int Rows, int Cols>
  decltype(auto) bottomLeftCorner(Index r) const {
    return block<Rows, Cols>(r, 0);
  }

  template <int Rows, int Cols>
  decltype(auto) bottomRightCorner(Index r, Index c) {
    return block<Rows, Cols>(r, c);
  }

  template <int Rows, int Cols>
  decltype(auto) bottomRightCorner(Index r, Index c) const {
    return block<Rows, Cols>(r, c);
  }

  // ---------------------------------------------------------------------------
  // Querying and Fast Block Vector / Index Access
  // ---------------------------------------------------------------------------

  template <int Rows, int Cols>
  [[nodiscard]] bool has_block(Index r, Index c) const noexcept {
    constexpr size_t type_r = find_type_index<Rows>();
    constexpr size_t type_c = find_type_index<Cols>();
    if constexpr (type_r <= type_c) {
      constexpr size_t P = internal::PairIndex(type_r, type_c, kNumBlockTypes);
      return std::get<P>(storages_).HasBlock(r, c);
    } else {
      constexpr size_t P = internal::PairIndex(type_c, type_r, kNumBlockTypes);
      return std::get<P>(storages_).HasBlock(c, r);
    }
  }

  template <int Rows, int Cols>
  [[nodiscard]] int block_index(Index r, Index c) const noexcept {
    constexpr size_t type_r = find_type_index<Rows>();
    constexpr size_t type_c = find_type_index<Cols>();
    if constexpr (type_r <= type_c) {
      constexpr size_t P = internal::PairIndex(type_r, type_c, kNumBlockTypes);
      return std::get<P>(storages_).GetIndex(r, c);
    } else {
      constexpr size_t P = internal::PairIndex(type_c, type_r, kNumBlockTypes);
      return std::get<P>(storages_).GetIndex(c, r);
    }
  }

  template <int Rows, int Cols>
  [[nodiscard]] const std::vector<Matrix<Scalar, Rows, Cols>>& blocks() const noexcept {
    constexpr size_t type_r = find_type_index<Rows>();
    constexpr size_t type_c = find_type_index<Cols>();
    static_assert(type_r <= type_c,
                  "blocks<Rows, Cols>() must be requested in upper-triangular order");
    constexpr size_t P = internal::PairIndex(type_r, type_c, kNumBlockTypes);
    return std::get<P>(storages_).blocks;
  }

  template <int Rows, int Cols>
  [[nodiscard]] std::vector<Matrix<Scalar, Rows, Cols>>& blocks() noexcept {
    constexpr size_t type_r = find_type_index<Rows>();
    constexpr size_t type_c = find_type_index<Cols>();
    static_assert(type_r <= type_c,
                  "blocks<Rows, Cols>() must be requested in upper-triangular order");
    constexpr size_t P = internal::PairIndex(type_r, type_c, kNumBlockTypes);
    return std::get<P>(storages_).blocks;
  }

  template <int Rows, int Cols>
  [[nodiscard]] const std::vector<BlockIndex>& block_indices() const noexcept {
    constexpr size_t type_r = find_type_index<Rows>();
    constexpr size_t type_c = find_type_index<Cols>();
    static_assert(type_r <= type_c,
                  "block_indices<Rows, Cols>() must be requested in upper-triangular order");
    constexpr size_t P = internal::PairIndex(type_r, type_c, kNumBlockTypes);
    return std::get<P>(storages_).coords;
  }

  // ---------------------------------------------------------------------------
  // Dense and Sparse Conversions
  // ---------------------------------------------------------------------------

  [[nodiscard]] Matrix<Scalar, Dynamic, Dynamic> toDense() const {
    Index tot = rows();
    Matrix<Scalar, Dynamic, Dynamic> mat = Matrix<Scalar, Dynamic, Dynamic>::Zero(tot, tot);

    auto add_storage = [&]<size_t K>(std::integral_constant<size_t, K>) {
      constexpr auto pr = internal::KToPair<kNumBlockTypes>(K);
      constexpr size_t type_i = pr.first;
      constexpr size_t type_j = pr.second;
      constexpr int Si = kBlockSizes[type_i];
      constexpr int Sj = kBlockSizes[type_j];
      const auto& st = std::get<K>(storages_);
      Index off_i = base_offset(type_i);
      Index off_j = base_offset(type_j);

      for (size_t k = 0; k < st.blocks.size(); ++k) {
        Index r = st.coords[k].row;
        Index c = st.coords[k].col;
        const auto& blk = st.blocks[k];
        Index r_start = off_i + r * Si;
        Index c_start = off_j + c * Sj;
        if constexpr (type_i == type_j) {
          if (r == c) {
            mat.template block<Si, Si>(r_start, c_start) = blk;
          } else {
            mat.template block<Si, Si>(r_start, c_start) = blk;
            mat.template block<Si, Si>(c_start, r_start) = blk.transpose();
          }
        } else {
          mat.template block<Si, Sj>(r_start, c_start) = blk;
          mat.template block<Sj, Si>(c_start, r_start) = blk.transpose();
        }
      }
    };

    [&]<size_t... Ks>(std::index_sequence<Ks...>) {
      (add_storage(std::integral_constant<size_t, Ks>{}), ...);
    }(std::make_index_sequence<kNumPairs>{});

    return mat;
  }

  [[nodiscard]] Matrix<Scalar, Dynamic, Dynamic> to_dense() const { return toDense(); }

  [[nodiscard]] SparseMatrix<Scalar> toSparse() const {
    Index tot = rows();
    SparseMatrix<Scalar> sp(tot, tot);
    std::vector<Eigen::Triplet<Scalar>> triplets;
    triplets.reserve(static_cast<size_t>(nonZeros()));

    auto add_storage = [&]<size_t K>(std::integral_constant<size_t, K>) {
      constexpr auto pr = internal::KToPair<kNumBlockTypes>(K);
      constexpr size_t type_i = pr.first;
      constexpr size_t type_j = pr.second;
      constexpr int Si = kBlockSizes[type_i];
      constexpr int Sj = kBlockSizes[type_j];
      const auto& st = std::get<K>(storages_);
      Index off_i = base_offset(type_i);
      Index off_j = base_offset(type_j);

      for (size_t k = 0; k < st.blocks.size(); ++k) {
        Index r = st.coords[k].row;
        Index c = st.coords[k].col;
        const auto& blk = st.blocks[k];
        Index r_start = off_i + r * Si;
        Index c_start = off_j + c * Sj;
        if constexpr (type_i == type_j) {
          if (r == c) {
            for (int u = 0; u < Si; ++u) {
              for (int v = 0; v < Si; ++v) {
                if (blk(u, v) != Scalar(0)) {
                  triplets.emplace_back(r_start + u, c_start + v, blk(u, v));
                }
              }
            }
          } else {
            for (int u = 0; u < Si; ++u) {
              for (int v = 0; v < Si; ++v) {
                if (blk(u, v) != Scalar(0)) {
                  triplets.emplace_back(r_start + u, c_start + v, blk(u, v));
                  triplets.emplace_back(c_start + v, r_start + u, blk(u, v));
                }
              }
            }
          }
        } else {
          for (int u = 0; u < Si; ++u) {
            for (int v = 0; v < Sj; ++v) {
              if (blk(u, v) != Scalar(0)) {
                triplets.emplace_back(r_start + u, c_start + v, blk(u, v));
                triplets.emplace_back(c_start + v, r_start + u, blk(u, v));
              }
            }
          }
        }
      }
    };

    [&]<size_t... Ks>(std::index_sequence<Ks...>) {
      (add_storage(std::integral_constant<size_t, Ks>{}), ...);
    }(std::make_index_sequence<kNumPairs>{});

    sp.setFromTriplets(triplets.begin(), triplets.end());
    return sp;
  }

  [[nodiscard]] SparseMatrix<Scalar> to_sparse() const { return toSparse(); }

  [[nodiscard]] BlockSparseMatX<Scalar> toBlockSparseMatX() const {
    std::vector<int> sizes(kBlockSizes, kBlockSizes + kNumBlockTypes);
    std::vector<Index> counts(kNumBlockTypes);
    for (size_t i = 0; i < kNumBlockTypes; ++i) {
      counts[i] = num_blocks(i);
    }
    return BlockSparseMatX<Scalar>(toSparse(), std::move(sizes), std::move(counts));
  }

  // ---------------------------------------------------------------------------
  // Algebraic Operations and Utilities
  // ---------------------------------------------------------------------------

  [[nodiscard]] Vector<Scalar, Dynamic> diagonal() const {
    Index tot = rows();
    Vector<Scalar, Dynamic> diag = Vector<Scalar, Dynamic>::Zero(tot);

    auto add_diag = [&]<size_t K>(std::integral_constant<size_t, K>) {
      constexpr auto pr = internal::KToPair<kNumBlockTypes>(K);
      if constexpr (pr.first == pr.second) {
        constexpr size_t type_i = pr.first;
        constexpr int Si = kBlockSizes[type_i];
        const auto& st = std::get<K>(storages_);
        Index off_i = base_offset(type_i);

        for (size_t k = 0; k < st.blocks.size(); ++k) {
          if (st.coords[k].row == st.coords[k].col) {
            Index r = st.coords[k].row;
            diag.template segment<Si>(off_i + r * Si) = st.blocks[k].diagonal();
          }
        }
      }
    };

    [&]<size_t... Ks>(std::index_sequence<Ks...>) {
      (add_diag(std::integral_constant<size_t, Ks>{}), ...);
    }(std::make_index_sequence<kNumPairs>{});

    return diag;
  }

  void scaleDiagonal(Scalar factor) {
    auto scale_diagonal = [&]<size_t K>(std::integral_constant<size_t, K>) {
      constexpr auto pr = internal::KToPair<kNumBlockTypes>(K);
      if constexpr (pr.first == pr.second) {
        auto& st = std::get<K>(storages_);
        for (size_t k = 0; k < st.blocks.size(); ++k) {
          if (st.coords[k].row == st.coords[k].col) {
            st.blocks[k].diagonal() *= factor;
          }
        }
      }
    };
    [&]<size_t... Ks>(std::index_sequence<Ks...>) {
      (scale_diagonal(std::integral_constant<size_t, Ks>{}), ...);
    }(std::make_index_sequence<kNumPairs>{});
  }

  void scaleByDiagonal(const Vector<Scalar, Dynamic>& scale, bool inverse = false) {
    auto scale_blocks = [&]<size_t K>(std::integral_constant<size_t, K>) {
      constexpr auto pr = internal::KToPair<kNumBlockTypes>(K);
      constexpr size_t row_type = pr.first;
      constexpr size_t col_type = pr.second;
      constexpr int block_rows = kBlockSizes[row_type];
      constexpr int block_cols = kBlockSizes[col_type];
      auto& st = std::get<K>(storages_);
      const Index row_offset = base_offset(row_type);
      const Index col_offset = base_offset(col_type);
      for (size_t k = 0; k < st.blocks.size(); ++k) {
        auto& block = st.blocks[k];
        const Index row_start = row_offset + st.coords[k].row * block_rows;
        const Index col_start = col_offset + st.coords[k].col * block_cols;
        const auto sr = scale.template segment<block_rows>(row_start);
        const auto sc = scale.template segment<block_cols>(col_start);
        if (inverse) {
          block.array() /= (sr * sc.transpose()).array();
        } else {
          block.array() *= (sr * sc.transpose()).array();
        }
      }
    };
    [&]<size_t... Ks>(std::index_sequence<Ks...>) {
      (scale_blocks(std::integral_constant<size_t, Ks>{}), ...);
    }(std::make_index_sequence<kNumPairs>{});
  }

  /// Apply fixed indices by zeroing their corresponding rows and columns, and setting
  /// diagonal elements to 1.
  ///
  /// @note fixed_indices MUST be provided in strictly increasing order.
  /// Any parameter dimension within any block type can be fixed.
  void applyFixedIndices(const std::vector<Index>& fixed_indices) {
    if (fixed_indices.empty()) return;

    // Track fixed blocks and bitmasks of fixed local coordinates for each block type.
    // Since fixed_indices is in increasing order, we scan block types and blocks sequentially.
    std::array<std::vector<uint32_t>, kNumBlockTypes> block_masks;
    std::array<std::vector<Index>, kNumBlockTypes> fixed_blocks;
    for (size_t type = 0; type < kNumBlockTypes; ++type) {
      block_masks[type].resize(static_cast<size_t>(num_blocks(type)), 0u);
    }

    size_t current_type = 0;
    Index current_offset = 0;
    Index next_offset = num_blocks(0) * kBlockSizes[0];

    for (const Index index : fixed_indices) {
      if (index < 0) continue;
      while (index >= next_offset && current_type + 1 < kNumBlockTypes) {
        current_offset = next_offset;
        ++current_type;
        next_offset += num_blocks(current_type) * kBlockSizes[current_type];
      }
      if (index < current_offset || index >= next_offset) continue;

      const int block_size = kBlockSizes[current_type];
      const Index block_idx = (index - current_offset) / block_size;
      const int local_idx = static_cast<int>((index - current_offset) % block_size);

      if (static_cast<size_t>(block_idx) < block_masks[current_type].size()) {
        if (block_masks[current_type][static_cast<size_t>(block_idx)] == 0u) {
          fixed_blocks[current_type].push_back(block_idx);
        }
        block_masks[current_type][static_cast<size_t>(block_idx)] |= (1u << local_idx);
      }
    }

    // 1. Update diagonal blocks in O(number of fixed blocks)
    auto set_diagonal_blocks = [&]<size_t Type>(std::integral_constant<size_t, Type>) {
      constexpr int S = kBlockSizes[Type];
      for (const Index b : fixed_blocks[Type]) {
        decltype(auto) block = block_by_type<Type, Type>(b, b);
        const uint32_t mask = block_masks[Type][static_cast<size_t>(b)];
        for (int local = 0; local < S; ++local) {
          if ((mask & (1u << local)) != 0u) {
            block.row(local).setZero();
            block.col(local).setZero();
            block(local, local) = Scalar(1);
          }
        }
      }
    };
    [&]<size_t... Types>(std::index_sequence<Types...>) {
      (set_diagonal_blocks(std::integral_constant<size_t, Types>{}), ...);
    }(std::make_index_sequence<kNumBlockTypes>{});

    // 2. Clear rows/columns in off-diagonal coupling blocks
    auto clear_off_diagonal = [&]<size_t K>(std::integral_constant<size_t, K>) {
      constexpr auto pr = internal::KToPair<kNumBlockTypes>(K);
      constexpr size_t T1 = pr.first;
      constexpr size_t T2 = pr.second;
      constexpr int rows = kBlockSizes[T1];
      constexpr int cols = kBlockSizes[T2];

      if (fixed_blocks[T1].empty() && fixed_blocks[T2].empty()) {
        return;  // Fast path: this storage pair has no fixed blocks at all
      }

      auto& st = std::get<K>(storages_);
      const auto& mask1 = block_masks[T1];
      const auto& mask2 = block_masks[T2];

      for (size_t k = 0; k < st.blocks.size(); ++k) {
        const Index r = st.coords[k].row;
        const Index c = st.coords[k].col;
        if constexpr (T1 == T2) {
          if (r == c) continue;  // Diagonal blocks handled above
        }

        const uint32_t m1 =
            (static_cast<size_t>(r) < mask1.size()) ? mask1[static_cast<size_t>(r)] : 0u;
        const uint32_t m2 =
            (static_cast<size_t>(c) < mask2.size()) ? mask2[static_cast<size_t>(c)] : 0u;

        if ((m1 | m2) == 0u) continue;  // Ultra-fast skip for unaffected blocks

        auto& block = st.blocks[k];
        if (m1 != 0u) {
          for (int row = 0; row < rows; ++row) {
            if ((m1 & (1u << row)) != 0u) {
              block.row(row).setZero();
            }
          }
        }
        if (m2 != 0u) {
          for (int col = 0; col < cols; ++col) {
            if ((m2 & (1u << col)) != 0u) {
              block.col(col).setZero();
            }
          }
        }
      }
    };
    [&]<size_t... Ks>(std::index_sequence<Ks...>) {
      (clear_off_diagonal(std::integral_constant<size_t, Ks>{}), ...);
    }(std::make_index_sequence<kNumPairs>{});
  }

  [[nodiscard]] Vector<Scalar, Dynamic> operator*(const Vector<Scalar, Dynamic>& x) const {
    Index tot = rows();
    Vector<Scalar, Dynamic> y = Vector<Scalar, Dynamic>::Zero(tot);

    auto mult_storage = [&]<size_t K>(std::integral_constant<size_t, K>) {
      constexpr auto pr = internal::KToPair<kNumBlockTypes>(K);
      constexpr size_t type_i = pr.first;
      constexpr size_t type_j = pr.second;
      constexpr int Si = kBlockSizes[type_i];
      constexpr int Sj = kBlockSizes[type_j];
      const auto& st = std::get<K>(storages_);
      Index off_i = base_offset(type_i);
      Index off_j = base_offset(type_j);

      for (size_t k = 0; k < st.blocks.size(); ++k) {
        Index r = st.coords[k].row;
        Index c = st.coords[k].col;
        const auto& blk = st.blocks[k];
        Index r_start = off_i + r * Si;
        Index c_start = off_j + c * Sj;
        if constexpr (type_i == type_j) {
          if (r == c) {
            y.template segment<Si>(r_start).noalias() += blk * x.template segment<Si>(r_start);
          } else {
            y.template segment<Si>(r_start).noalias() += blk * x.template segment<Si>(c_start);
            y.template segment<Si>(c_start).noalias() +=
                blk.transpose() * x.template segment<Si>(r_start);
          }
        } else {
          y.template segment<Si>(r_start).noalias() += blk * x.template segment<Sj>(c_start);
          y.template segment<Sj>(c_start).noalias() +=
              blk.transpose() * x.template segment<Si>(r_start);
        }
      }
    };

    [&]<size_t... Ks>(std::index_sequence<Ks...>) {
      (mult_storage(std::integral_constant<size_t, Ks>{}), ...);
    }(std::make_index_sequence<kNumPairs>{});

    return y;
  }

  [[nodiscard]] Scalar squaredNorm() const noexcept {
    Scalar total = 0;
    auto sq_norm = [&]<size_t K>(std::integral_constant<size_t, K>) {
      constexpr auto pr = internal::KToPair<kNumBlockTypes>(K);
      const auto& st = std::get<K>(storages_);
      if constexpr (pr.first == pr.second) {
        for (size_t k = 0; k < st.blocks.size(); ++k) {
          if (st.coords[k].row == st.coords[k].col) {
            total += st.blocks[k].squaredNorm();
          } else {
            total += Scalar(2) * st.blocks[k].squaredNorm();
          }
        }
      } else {
        for (size_t k = 0; k < st.blocks.size(); ++k) {
          total += Scalar(2) * st.blocks[k].squaredNorm();
        }
      }
    };
    [&]<size_t... Ks>(std::index_sequence<Ks...>) {
      (sq_norm(std::integral_constant<size_t, Ks>{}), ...);
    }(std::make_index_sequence<kNumPairs>{});
    return total;
  }

  [[nodiscard]] Scalar norm() const noexcept {
    using std::sqrt;
    return sqrt(squaredNorm());
  }

  [[nodiscard]] const BlockSparseMatrix& transpose() const noexcept { return *this; }

  [[nodiscard]] bool isApprox(const BlockSparseMatrix& other,
                              Scalar prec = Eigen::NumTraits<Scalar>::dummy_precision()) const {
    return (toDense() - other.toDense()).isZero(prec);
  }

 private:
  StorageTuple storages_;
  mutable std::array<Index, kNumBlockTypes> block_counts_{};
};

/// Alias matching common user spelling variation
template <typename Scalar, int... BlockSizes>
using BlockSparseMatix = BlockSparseMatrix<Scalar, BlockSizes...>;

template <int... BlockSizes>
using BlockSparseMat = BlockSparseMatrix<double, BlockSizes...>;

template <int... BlockSizes>
using BlockSparseMatf = BlockSparseMatrix<float, BlockSizes...>;

namespace traits {
template <typename Scalar, int... Sizes>
struct is_block_sparse_matrix<BlockSparseMatrix<Scalar, Sizes...>> : std::true_type {};
}  // namespace traits

}  // namespace tinyopt
