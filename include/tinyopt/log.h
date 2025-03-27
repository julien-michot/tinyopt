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