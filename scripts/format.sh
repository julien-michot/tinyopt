#!/usr/bin/env bash
# Copyright 2026 Julien Michot.
# SPDX-License-Identifier: Apache-2.0
#
# Helper script to run clang-format across all Tinyopt C++ headers and tests.
# Usage:
#   ./format.sh         # In-place format all .h and .cpp files
#   ./format.sh --check # Dry-run check for formatting discrepancies (exits with error if diff found)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${WORKSPACE_ROOT}"

CHECK_ONLY=false
if [[ "${1:-}" == "--check" ]]; then
    CHECK_ONLY=true
fi

# Find all C++ header and source files in the relevant project directories.
# Some repos may not have an examples/ directory, so skip missing folders.
FILES=()
for dir in include tests benchmarks examples; do
    if [[ -d "${dir}" ]]; then
        while IFS= read -r file; do
            FILES+=("${file}")
        done < <(find "${dir}" -type f \( -name "*.h" -o -name "*.cpp" -o -name "*.hpp" \) ! -path "*/.*")
    fi
done

if [[ "${CHECK_ONLY}" == "true" ]]; then
    echo "==> Verifying code formatting against .clang-format..."
    FAILED=0
    for f in ${FILES}; do
        if ! clang-format --dry-run --Werror "$f" > /dev/null 2>&1; then
            echo "Formatting error: $f"
            FAILED=1
        fi
    done
    if [[ ${FAILED} -ne 0 ]]; then
        echo "Error: Formatting violations found. Run './format.sh' to fix them automatically."
        exit 1
    fi
    echo "==> All files conform to .clang-format!"
else
    echo "==> Formatting C++ files in-place using clang-format..."
    for f in ${FILES}; do
        clang-format -i "$f"
    done
    echo "==> Formatting completed!"
fi
