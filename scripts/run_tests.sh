#!/usr/bin/env bash
# Copyright 2026 Julien Michot.
# SPDX-License-Identifier: Apache-2.0
#
# Helper script to clean, configure, compile, and run Tinyopt unit tests
# Usage:
#   ./run_tests.sh              # Run full test suite via ctest
#   ./run_tests.sh <test_name>  # Run a specific test executable (e.g. tinyopt_test_sqrt2)
#   ./run_tests.sh --clean      # Force a clean re-configure first

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${WORKSPACE_ROOT}"

TARGET="${1:-all}"
CLEAN_BUILD=false

if [[ "${TARGET}" == "--clean" ]]; then
    CLEAN_BUILD=true
    TARGET="${2:-all}"
fi

if [[ "${CLEAN_BUILD}" == "true" ]] || [[ ! -f "build/build.ninja" ]] || \
   grep -q '/bench/' build/CMakeCache.txt 2>/dev/null || \
   grep -q '/python/' build/CMakeCache.txt 2>/dev/null || \
   ! grep -q 'TINYOPT_BUILD_TESTS:BOOL=ON' build/CMakeCache.txt 2>/dev/null; then
    echo "==> Cleaning and configuring build in the 'test' Pixi environment..."
    pixi run configure-tests
fi

echo "==> Building test targets with Ninja..."
pixi run build-tests

if [[ "${TARGET}" == "all" ]]; then
    echo "==> Running full test suite via CTest..."
    cd build
    ctest --output-on-failure --verbose
else
    TEST_BIN="$(basename "${TARGET}")"
    if [[ ! -f "build/tests/${TEST_BIN}" ]]; then
        echo "Error: Test executable build/tests/${TEST_BIN} does not exist!"
        echo "Available tests in build/tests/:"
        ls -1 build/tests/tinyopt_test_* || true
        exit 1
    fi
    echo "==> Running individual test: build/tests/${TEST_BIN} with verbose Catch2 output..."
    "./build/tests/${TEST_BIN}" -s
fi

echo "==> All tests completed successfully!"
