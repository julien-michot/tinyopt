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

#if HAS_SPDLOG

#include <spdlog/fmt/fmt.h>
#if defined(SPDLOG_USE_STD_FORMAT)
#define TINYOPT_FORMAT std::format
#else
#define TINYOPT_FORMAT fmt::format
#endif
#define FMT_NAMESPACE fmt

#elif HAS_FMT
#include <fmt/color.h>
#include <fmt/compile.h>
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