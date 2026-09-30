// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifndef TINYOPT_FORMAT_NS
#ifdef HAS_FMT
#include <fmt/core.h>
#include <fmt/ostream.h>
#define TINYOPT_FORMAT_NS fmt
#else
#include <format>
#define TINYOPT_FORMAT_NS std
#endif
#endif

#ifndef TINYOPT_LOG
#include <iostream>
#define TINYOPT_LOG(...) std::cout << TINYOPT_FORMAT_NS::format(__VA_ARGS__) << std::endl;
#endif

#define TINYOPT_LOG_MAT(m) \
  TINYOPT_LOG("{}:{}x{}{}{}", #m, m.rows(), m.cols(), m.cols() == 1 ? "" : "\n", m);
// Include formatters
#ifndef TINYOPT_NO_FORMATTERS
#include "tinyopt/formatters.h"
#endif  // TINYOPT_NO_FORMATTERS
