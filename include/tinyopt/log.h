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

#include <sstream>

#include "traits.h"


#if HAS_FMT
#include <fmt/core.h>
#include <fmt/ostream.h>

#define TINYOPT_FORMAT fmt::format
#define FMT_NAMESPACE fmt

#elif __cplusplus >= 202002L
#include <format>
#define TINYOPT_FORMAT std::format

#elif !defined(TINYOPT_FORMAT)

#define TINYOPT_FORMAT(str, ...) str // c++ < 2020 not well supported for now

#endif

namespace tinyopt {

// Default toString
template <typename T>
std::string toString(const T& value) {
  std::stringstream ss;
  ss << value;
  return ss.str();
}

// Specialization for Eigen Matrix
template <typename T>
std::string toString(const Eigen::MatrixBase<T> &m) {
  std::stringstream ss;
  if (m.cols() == 1)
    ss << m.transpose();
  else
    ss << m;
  return ss.str();
};

} // namespace tinyopt