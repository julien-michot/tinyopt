// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <utility>

namespace tinyopt {

template <typename T, std::size_t Capacity>
class FixedFifo {
  static_assert(Capacity > 0);

 public:
  template <typename Value>
  void emplace_back(Value &&value) {
    if (size_ < Capacity) {
      values_[(start_ + size_) % Capacity] = std::forward<Value>(value);
      ++size_;
    } else {
      values_[start_] = std::forward<Value>(value);
      start_ = (start_ + 1) % Capacity;
    }
  }

  bool empty() const { return size_ == 0; }
  std::size_t size() const { return size_; }

  const T &back() const { return values_[(start_ + size_ - 1) % Capacity]; }
  T &back() { return values_[(start_ + size_ - 1) % Capacity]; }

 private:
  std::array<T, Capacity> values_{};
  std::size_t start_ = 0;
  std::size_t size_ = 0;
};

}  // namespace tinyopt