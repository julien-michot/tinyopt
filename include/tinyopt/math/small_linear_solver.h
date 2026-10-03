// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <optional>

#include <tinyopt/types.h>

namespace tinyopt {

template <typename Derived, typename Derived2>
std::optional<Vector<typename Derived::Scalar, Derived::RowsAtCompileTime>> SolveSmallDenseSystem(
    const MatrixBase<Derived> &A, const MatrixBase<Derived2> &b) {
  using Scalar = typename Derived::Scalar;
  using Result = Vector<Scalar, Derived::RowsAtCompileTime>;
  constexpr int Dims = Derived::RowsAtCompileTime;
  const Scalar epsilon = Eigen::NumTraits<Scalar>::epsilon();

  if constexpr (Dims == 1) {
    const Scalar value = A(0, 0);
    if (!(value > epsilon) || !Eigen::numext::isfinite(value)) return std::nullopt;
    Result solution = b / value;
    return solution;
  } else if constexpr (Dims == 2) {
    const Scalar a00 = A(0, 0);
    const Scalar a01 = A(0, 1);
    const Scalar a11 = A(1, 1);
    const Scalar trace = a00 + a11;
    const Scalar scale = std::max(Eigen::numext::abs(a00),
                                  std::max(Eigen::numext::abs(a01), Eigen::numext::abs(a11)));
    if (!(trace > epsilon * scale) || !Eigen::numext::isfinite(trace) ||
        !Eigen::numext::isfinite(scale))
      return std::nullopt;
    const Scalar n00 = a00 / trace;
    const Scalar n01 = a01 / trace;
    const Scalar n11 = a11 / trace;
    const Scalar determinant = n00 * n11 - n01 * n01;
    if (!(Eigen::numext::abs(determinant) > epsilon) || !Eigen::numext::isfinite(determinant))
      return std::nullopt;
    const Scalar denominator = trace * determinant;
    if (denominator == Scalar(0) || !Eigen::numext::isfinite(denominator)) return std::nullopt;
    Result solution;
    solution(0) = (n11 * b(0) - n01 * b(1)) / denominator;
    solution(1) = (n00 * b(1) - n01 * b(0)) / denominator;
    return solution;
  } else if constexpr (Dims == 3) {
    const Scalar a00 = A(0, 0);
    const Scalar a01 = A(0, 1);
    const Scalar a02 = A(0, 2);
    const Scalar a11 = A(1, 1);
    const Scalar a12 = A(1, 2);
    const Scalar a22 = A(2, 2);
    const Scalar trace = a00 + a11 + a22;
    const Scalar scale =
        std::max(std::max(Eigen::numext::abs(a00), Eigen::numext::abs(a01)),
                 std::max(std::max(Eigen::numext::abs(a02), Eigen::numext::abs(a11)),
                          std::max(Eigen::numext::abs(a12), Eigen::numext::abs(a22))));
    if (!(Eigen::numext::abs(trace) > epsilon * scale) || !Eigen::numext::isfinite(trace) ||
        !Eigen::numext::isfinite(scale))
      return std::nullopt;
    const Scalar n00 = a00 / trace;
    const Scalar n01 = a01 / trace;
    const Scalar n02 = a02 / trace;
    const Scalar n11 = a11 / trace;
    const Scalar n12 = a12 / trace;
    const Scalar n22 = a22 / trace;
    const Scalar determinant = n00 * (n11 * n22 - n12 * n12) - n01 * (n01 * n22 - n12 * n02) +
                               n02 * (n01 * n12 - n11 * n02);
    if (!(Eigen::numext::abs(determinant) > epsilon) || !Eigen::numext::isfinite(determinant))
      return std::nullopt;
    const Scalar denominator = trace * determinant;
    if (denominator == Scalar(0) || !Eigen::numext::isfinite(denominator)) return std::nullopt;
    Result solution;
    solution(0) = ((n11 * n22 - n12 * n12) * b(0) + (n02 * n12 - n01 * n22) * b(1) +
                   (n01 * n12 - n02 * n11) * b(2)) /
                  denominator;
    solution(1) = ((n02 * n12 - n01 * n22) * b(0) + (n00 * n22 - n02 * n02) * b(1) +
                   (n01 * n02 - n00 * n12) * b(2)) /
                  denominator;
    solution(2) = ((n01 * n12 - n02 * n11) * b(0) + (n01 * n02 - n00 * n12) * b(1) +
                   (n00 * n11 - n01 * n01) * b(2)) /
                  denominator;
    return solution;
  } else {
    static_assert(Dims == 4);
    const Scalar a00 = A(0, 0);
    const Scalar a01 = A(0, 1);
    const Scalar a02 = A(0, 2);
    const Scalar a03 = A(0, 3);
    const Scalar a11 = A(1, 1);
    const Scalar a12 = A(1, 2);
    const Scalar a13 = A(1, 3);
    const Scalar a22 = A(2, 2);
    const Scalar a23 = A(2, 3);
    const Scalar a33 = A(3, 3);
    const Scalar trace = a00 + a11 + a22 + a33;
    const Scalar scale =
        std::max(std::max(std::max(Eigen::numext::abs(a00), Eigen::numext::abs(a01)),
                          std::max(Eigen::numext::abs(a02), Eigen::numext::abs(a03))),
                 std::max(std::max(Eigen::numext::abs(a11), Eigen::numext::abs(a12)),
                          std::max(std::max(Eigen::numext::abs(a13), Eigen::numext::abs(a22)),
                                   std::max(Eigen::numext::abs(a23), Eigen::numext::abs(a33)))));
    if (!(Eigen::numext::abs(trace) > epsilon * scale) || !Eigen::numext::isfinite(trace) ||
        !Eigen::numext::isfinite(scale))
      return std::nullopt;

    const Scalar n00 = a00 / trace;
    const Scalar n01 = a01 / trace;
    const Scalar n02 = a02 / trace;
    const Scalar n03 = a03 / trace;
    const Scalar n11 = a11 / trace;
    const Scalar n12 = a12 / trace;
    const Scalar n13 = a13 / trace;
    const Scalar n22 = a22 / trace;
    const Scalar n23 = a23 / trace;
    const Scalar n33 = a33 / trace;
    const auto det3 = [](Scalar m00, Scalar m01, Scalar m02, Scalar m10, Scalar m11, Scalar m12,
                         Scalar m20, Scalar m21, Scalar m22) {
      return m00 * (m11 * m22 - m12 * m21) - m01 * (m10 * m22 - m12 * m20) +
             m02 * (m10 * m21 - m11 * m20);
    };
    const Scalar c00 = det3(n11, n12, n13, n12, n22, n23, n13, n23, n33);
    const Scalar c01 = -det3(n01, n12, n13, n02, n22, n23, n03, n23, n33);
    const Scalar c02 = det3(n01, n11, n13, n02, n12, n23, n03, n13, n33);
    const Scalar c03 = -det3(n01, n11, n12, n02, n12, n22, n03, n13, n23);
    const Scalar determinant = n00 * c00 + n01 * c01 + n02 * c02 + n03 * c03;
    if (!(Eigen::numext::abs(determinant) > epsilon * epsilon) ||
        !Eigen::numext::isfinite(determinant))
      return std::nullopt;
    const Scalar denominator = trace * determinant;
    if (denominator == Scalar(0) || !Eigen::numext::isfinite(denominator)) return std::nullopt;
    const Scalar c11 = det3(n00, n02, n03, n02, n22, n23, n03, n23, n33);
    const Scalar c12 = -det3(n00, n01, n03, n02, n12, n23, n03, n13, n33);
    const Scalar c13 = det3(n00, n01, n02, n02, n12, n22, n03, n13, n23);
    const Scalar c22 = det3(n00, n01, n03, n01, n11, n13, n03, n13, n33);
    const Scalar c23 = -det3(n00, n01, n02, n01, n11, n12, n03, n13, n23);
    const Scalar c33 = det3(n00, n01, n02, n01, n11, n12, n02, n12, n22);
    Result solution;
    solution(0) = (c00 * b(0) + c01 * b(1) + c02 * b(2) + c03 * b(3)) / denominator;
    solution(1) = (c01 * b(0) + c11 * b(1) + c12 * b(2) + c13 * b(3)) / denominator;
    solution(2) = (c02 * b(0) + c12 * b(1) + c22 * b(2) + c23 * b(3)) / denominator;
    solution(3) = (c03 * b(0) + c13 * b(1) + c23 * b(2) + c33 * b(3)) / denominator;
    return solution;
  }
}

}  // namespace tinyopt