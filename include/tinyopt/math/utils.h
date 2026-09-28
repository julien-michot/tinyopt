// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <type_traits>

#include <tinyopt/types.h>

namespace tinyopt {

/// Integer square root function for positive integers. Returns N for negative or zero values.
constexpr inline int SQRT(int N) {
  if (N <= 1) return N;
  int left = 1, right = N / 2;
  int result = 0;
  while (left <= right) {
    int mid = left + (right - left) / 2;
    if (mid <= N / mid) {
      left = mid + 1;
      result = mid;
    } else {
      right = mid - 1;
    }
  }
  return result;
};

template <typename Scalar = double>
inline constexpr Scalar FloatEpsilon() {
  /*static*/ const Scalar eps = static_cast<Scalar>(std::is_same_v<Scalar, float> ? 1e-4f : 1e-7f);
  return eps;
}

template <typename Scalar = double>
inline constexpr Scalar FloatEpsilon2() {
  /*static*/ const Scalar eps = static_cast<Scalar>(std::is_same_v<Scalar, float> ? 1e-8f : 1e-14f);
  return eps;
}

/// A constexpr version of the ternary operator: (condition) ? ValueOnTrue : ValueOnFalse
#ifndef _MSC_VER  // due to error C3493...
#define If(condition, ValueOnTrue, ValueOnFalse) \
  [&]() {                                        \
    if constexpr (condition)                     \
      return ValueOnTrue;                        \
    else                                         \
      return ValueOnFalse;                       \
  }()
#endif

template <typename T>
T MaxAbsDiff(const Eigen::SparseMatrix<T> &mat1, const Eigen::SparseMatrix<T> &mat2) {
  if (mat1.rows() != mat2.rows() || mat1.cols() != mat2.cols()) {
    throw std::invalid_argument("Matrices must have the same dimensions.");
  }

  T maxDiff = 0;
  for (int k = 0; k < mat1.outerSize(); ++k) {
    for (typename Eigen::SparseMatrix<T>::InnerIterator it1(mat1, k); it1; ++it1) {
      T val2 = 0;
      for (typename Eigen::SparseMatrix<T>::InnerIterator it2(mat2, k); it2; ++it2) {
        if (it2.row() == it1.row()) {
          val2 = it2.value();
          break;
        }
      }
      maxDiff = std::max(maxDiff, std::abs(it1.value() - val2));
    }
  }

  for (int k = 0; k < mat2.outerSize(); ++k) {
    for (typename Eigen::SparseMatrix<T>::InnerIterator it2(mat2, k); it2; ++it2) {
      T val1 = 0;
      bool found = false;
      for (typename Eigen::SparseMatrix<T>::InnerIterator it1(mat1, k); it1; ++it1) {
        if (it1.row() == it2.row()) {
          val1 = it1.value();
          found = true;
          break;
        }
      }
      if (!found) maxDiff = std::max(maxDiff, std::abs(val1 - it2.value()));
    }
  }
  return maxDiff;
}

}  // namespace tinyopt