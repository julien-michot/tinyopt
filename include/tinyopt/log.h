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
#ifdef TINYOPT_FORMAT
// externally defined

#elif HAS_FMT
#include <fmt/core.h>
#include <fmt/ostream.h>

#define TINYOPT_FORMAT fmt::format
#define TINYOPT_FORMAT_NAMESPACE fmt

#elif __cplusplus >= 202002L

#include <format>
#define TINYOPT_FORMAT std::format
#define TINYOPT_FORMAT_NAMESPACE std

#endif

// Include formatters
#ifndef TINYOPT_NO_FORMATTERS
#include "tinyopt/formatters.h"
#endif  // TINYOPT_NO_FORMATTERS

namespace tinyopt::log {

/// Logging struct
struct Logging {
  class SilencePlease : public std::streambuf {
   public:
    int_type overflow(int_type c) override {
      return traits_type::not_eof(c);  // Indicate success.
    }
    std::streamsize xsputn(const char *, std::streamsize n) override {
      return n;  // Indicate that all characters were "written".
    }
  };
  std::ostream &oss = std::cout;  ///< Stream used for logging

  /// Disable logging
  void Disable() {
    static SilencePlease silence;
    oss.rdbuf(&silence);
  }

  /// Enable logging on a given stream (default is std::cout)
  void Enable(std::ostream &stream = std::cout) { oss.rdbuf(stream.rdbuf()); }
};
}  // namespace tinyopt::log