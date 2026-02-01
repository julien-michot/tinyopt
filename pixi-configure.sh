#!/bin/bash
set -e

CMAKE_FLAGS="$*"

# Detect Catch2
if [ -d "$CONDA_PREFIX/include/catch2" ]; then
  echo "Catch2 dependency found, enabling tests."
  CMAKE_FLAGS="$CMAKE_FLAGS -DTINYOPT_BUILD_TESTS=ON"
fi

# Detect Ceres
if [ -d "$CONDA_PREFIX/include/ceres" ]; then
  echo "Ceres dependency found, enabling benchmarks and ceres tests."
  CMAKE_FLAGS="$CMAKE_FLAGS -DTINYOPT_BUILD_BENCHMARKS=ON -DTINYOPT_BUILD_CERES=ON"
fi

# Detect Sophus
if [ -d "$CONDA_PREFIX/include/sophus" ]; then
  echo "Sophus dependency found, enabling Sophus tests."
  CMAKE_FLAGS="$CMAKE_FLAGS -DTINYOPT_BUILD_SOPHUS_TEST=ON"
fi

# Detect Doxygen
if [ -f "$CONDA_PREFIX/bin/doxygen" ]; then
  echo "Doxygen dependency found, enabling docs."
  CMAKE_FLAGS="$CMAKE_FLAGS -DTINYOPT_BUILD_DOCS=ON"
fi

echo "Configuring with flags: $CMAKE_FLAGS"

# FIX: Call cmake directly or use the pixi task without the '--' separator
# Option A: Call cmake directly (most reliable for script logic)
cmake -B build -G Ninja $CMAKE_FLAGS