// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>

namespace {

int log_calls = 0;

template <typename... Args>
void RecordLog(Args &&...) {
  ++log_calls;
}

}  // namespace

#define TINYOPT_LOG(...) RecordLog(__VA_ARGS__)
#include <tinyopt/log.h>

TEST_CASE("externally defined logger receives Tinyopt log output") {
  Eigen::Matrix2d matrix = Eigen::Matrix2d::Identity();
  log_calls = 0;

  TINYOPT_LOG("message {}", 1);
  TINYOPT_LOG_MAT(matrix);

  REQUIRE(log_calls == 2);
}