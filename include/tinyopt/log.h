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

#include <iostream>

#ifdef TINYOPT_LOG

// TINYOPT_LOG(...) is externally defined so we'll use it

#elif HAS_FMT
#include <fmt/core.h>
#include <fmt/ostream.h>

#define TINYOPT_LOG(...) fmt::print(__VA_ARGS__);
#define TINYOPT_FORMAT_NAMESPACE fmt

#elif __cplusplus >= 202002L

#include <format>

#define TINYOPT_LOG(...) std::cout << std::format(__VA_ARGS__) << std::endl;
#define TINYOPT_FORMAT_NAMESPACE std

#else  // c++ 17 and below

#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <tinyopt/traits.h>

// Add 'dummy' tinyopt::format

namespace tinyopt {

/// Dummy function that replaces {*} with the arg. Does not support formatting as such!
std::string format2(const std::string &format_string, const std::vector<std::string> &args) {
  std::stringstream result;
  size_t arg_index = 0;

  for (size_t i = 0; i < format_string.length(); ++i) {
    if (format_string[i] == '{') {
      if (arg_index >= args.size()) {
        throw std::out_of_range("Not enough arguments for format string.");
      }
      result << args[arg_index++];
      // Skip until '}'
      while (format_string[++i] != '}' && i < format_string.size()) {
      }
    } else if (format_string[i] == '}') {
      throw std::invalid_argument("Invalid format string.");
    } else {
      result << format_string[i];
    }
  }

  if (arg_index < args.size()) {
    throw std::invalid_argument("Too many arguments for format string.");
  }

  return result.str();
}

/// Dummy function that replaces {*} with the arg. Does not support formatting as such!
template <typename... Args>
std::string format(const std::string &format_string, Args &&...args) {
  std::vector<std::string> arg_strings;
  std::ostringstream converter;
  auto add_arg = [&](auto &&arg) {  // Lambda to avoid comma in fold
    converter.str("");
    if constexpr (tinyopt::traits::is_streamable_v<decltype(arg)>)
      converter << std::forward<decltype(arg)>(arg);
    arg_strings.push_back(converter.str());
  };
  (add_arg(std::forward<Args>(args)), ...);  // Correct fold expression

  (void)add_arg;

  return format2(format_string, arg_strings);
}
}  // namespace tinyopt

#define TINYOPT_LOG(...) std::cout << tinyopt::format(__VA_ARGS__) << std::endl;
#define TINYOPT_FORMAT_NAMESPACE tinyopt

#endif

#define TINYOPT_LOG_MAT(m)                                                              \
  std::cout << TINYOPT_FORMAT_NAMESPACE::format("{}:{}x{}{}{}", #m, m.rows(), m.cols(), \
                                                m.cols() == 1 ? "" : "\n", m)           \
            << std::endl;
// Include formatters
#ifndef TINYOPT_NO_FORMATTERS
#include "tinyopt/formatters.h"
#endif  // TINYOPT_NO_FORMATTERS
