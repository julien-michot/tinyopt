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

#include <format>
#include <sstream>

#include <tinyopt/math.h>

#ifdef TINYOPT_FORMAT_NS

#include <tinyopt/traits.h>

#if __cplusplus >= 202002L

template <typename T>
struct TINYOPT_FORMAT_NS::formatter<
    T, std::enable_if_t<std::is_base_of_v<tinyopt::DenseBase<T>, T>, char>> {
  template <typename ParseContext>
  constexpr auto parse(ParseContext& ctx) {
    return m_underlying.parse(ctx);
  }
  template <typename FormatContext>
  auto format(const T& m, FormatContext& ctx) const {
    std::ostringstream os;
    if (m.cols() == 1)  // print vectors as a row
      os << m.transpose();
    else
      os << m;
    auto out = ctx.out();
    out = TINYOPT_FORMAT_NS::format_to(ctx.out(), "{}", os.str());
    return out;
  }

 private:
  TINYOPT_FORMAT_NS::formatter<typename T::Scalar, char> m_underlying;
};

template <typename T>
struct TINYOPT_FORMAT_NS::formatter<
    T, std::enable_if_t<tinyopt::traits::is_sparse_matrix_v<T>, char>> {
  template <typename ParseContext>
  constexpr auto parse(ParseContext& ctx) {
    return m_underlying.parse(ctx);
  }
  template <typename FormatContext>
  auto format(const T& m, FormatContext& ctx) const {
    std::ostringstream os;
    os << m;
    auto out = ctx.out();
    out = TINYOPT_FORMAT_NS::format_to(ctx.out(), "{}", os.str());
    return out;
  }

 private:
  TINYOPT_FORMAT_NS::formatter<typename T::Scalar, char> m_underlying;
};

#endif

#if __cplusplus < 202302L

// std::array
template <typename T, size_t N>
std::ostream& operator<<(std::ostream& os, const std::array<T, N>& v) {
  for (std::size_t i = 0; i < v.size(); ++i) {
    os << v[i];
    if (i + 1 < v.size()) os << " ";
  }
  return os;
}

// std::vector
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v) {
  for (std::size_t i = 0; i < v.size(); ++i) {
    os << v[i];
    if (i + 1 < v.size()) os << " ";
  }
  return os;
}

#endif  // c++23

#endif
