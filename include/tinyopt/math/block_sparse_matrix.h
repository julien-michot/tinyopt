// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <tuple>
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

  // Direct 2D lookup table: index_grid[r * grid_cols + c] gives index in blocks, or -1.
  std::vector<int> index_grid;
  Index grid_rows = 0;
  Index grid_cols = 0;

  void EnsureGrid(Index r, Index c) {
    if (r >= grid_rows || c >= grid_cols) {
      Index new_rows = std::max(r + 1, grid_rows == 0 ? Index(4) : grid_rows * 2);
      Index new_cols = std::max(c + 1, grid_cols == 0 ? Index(4) : grid_cols * 2);
      std::vector<int> new_grid(static_cast<size_t>(new_rows * new_cols), -1);
      for (size_t k = 0; k < coords.size(); ++k) {
        new_grid[static_cast<size_t>(coords[k].row * new_cols + coords[k].col)] =
            static_cast<int>(k);
      }
      index_grid = std::move(new_grid);
      grid_rows = new_rows;
      grid_cols = new_cols;
    }
  }

  [[nodiscard]] int GetIndex(Index r, Index c) const noexcept {
    if (r < grid_rows && c < grid_cols) {
      return index_grid[static_cast<size_t>(r * grid_cols + c)];
    }
    return -1;
  }

  [[nodiscard]] bool HasBlock(Index r, Index c) const noexcept { return GetIndex(r, c) != -1; }

  MatrixType& GetOrCreateBlock(Index r, Index c) {
    int idx = GetIndex(r, c);
    if (idx != -1) {
      return blocks[static_cast<size_t>(idx)];
    }
    EnsureGrid(r, c);
    int new_idx = static_cast<int>(blocks.size());
    blocks.emplace_back(MatrixType::Zero());
    coords.emplace_back(r, c);
    index_grid[static_cast<size_t>(r * grid_cols + c)] = new_idx;
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
    blocks.clear();
    coords.clear();
    std::fill(index_grid.begin(), index_grid.end(), -1);
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
        for (int col = 0; col < block_cols; ++col) {
          for (int row = 0; row < block_rows; ++row) {
            const Scalar factor = scale[row_start + row] * scale[col_start + col];
            if (inverse)
              block(row, col) /= factor;
            else
              block(row, col) *= factor;
          }
        }
      }
    };
    [&]<size_t... Ks>(std::index_sequence<Ks...>) {
      (scale_blocks(std::integral_constant<size_t, Ks>{}), ...);
    }(std::make_index_sequence<kNumPairs>{});
  }

  void applyLockedIndices(const std::vector<Index>& locked_indices) {
    const Index dimension = rows();
    std::vector<bool> locked(static_cast<size_t>(dimension), false);
    for (const Index index : locked_indices) {
      if (index >= 0 && index < dimension) locked[static_cast<size_t>(index)] = true;
    }

    auto clear_locked_entries = [&]<size_t K>(std::integral_constant<size_t, K>) {
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
        for (int col = 0; col < block_cols; ++col) {
          for (int row = 0; row < block_rows; ++row) {
            if (locked[static_cast<size_t>(row_start + row)] ||
                locked[static_cast<size_t>(col_start + col)]) {
              block(row, col) = Scalar(0);
            }
          }
        }
      }
    };
    [&]<size_t... Ks>(std::index_sequence<Ks...>) {
      (clear_locked_entries(std::integral_constant<size_t, Ks>{}), ...);
    }(std::make_index_sequence<kNumPairs>{});

    auto set_locked_diagonal = [&]<size_t Type>(std::integral_constant<size_t, Type>) {
      constexpr int block_size = kBlockSizes[Type];
      const Index offset = base_offset(Type);
      const Index count = num_blocks(Type);
      for (Index block = 0; block < count; ++block) {
        for (int local = 0; local < block_size; ++local) {
          const Index index = offset + block * block_size + local;
          if (index >= dimension || !locked[static_cast<size_t>(index)]) continue;
          block_by_type<Type, Type>(block, block)(local, local) = Scalar(1);
        }
      }
    };
    [&]<size_t... Types>(std::index_sequence<Types...>) {
      (set_locked_diagonal(std::integral_constant<size_t, Types>{}), ...);
    }(std::make_index_sequence<kNumBlockTypes>{});
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
