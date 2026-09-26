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

# If caller explicitly disabled Sophus tests, don't auto-enable them.
if echo "$CMAKE_FLAGS" | grep -q "TINYOPT_BUILD_SOPHUS_TEST=OFF"; then
  # Remove any previously-added ON flag (including from the auto-detection above).
  CMAKE_FLAGS=$(echo "$CMAKE_FLAGS" | sed -E 's/-DTINYOPT_BUILD_SOPHUS_TEST=ON//g')
fi

# Detect Doxygen
if [ -f "$CONDA_PREFIX/bin/doxygen" ]; then
  echo "Doxygen dependency found, enabling docs."
  CMAKE_FLAGS="$CMAKE_FLAGS -DTINYOPT_BUILD_DOCS=ON"
fi

echo "Configuring with flags: $CMAKE_FLAGS"

# If Python bindings are requested, try to ensure the Python 'clang' package
# (clang.cindex) is available in the active Python environment and try to
# auto-detect a libclang shared library on macOS (Homebrew). This helps when
# `pixi run configure-python` is used: pixi will create the environment and
# then this script will install the pip package into that environment.
if echo "$CMAKE_FLAGS" | grep -q "TINYOPT_BUILD_BINDINGS=ON"; then
  echo "$(which python3) $(which python)"
  echo "Bindings requested: checking for clang python package (used by the Python binding generator)..."
  # Use the python available on PATH (pixi runs commands inside the env)
  if python3 -c "import clang" >/dev/null 2>&1; then
    echo "clang python bindings already present"
  else
    echo "clang python bindings not found — installing via pip into current Python"

    # Try installing clang into the active environment (no --user).
    if python3 -m pip install clang; then
      echo "Installed clang python package"
    else
      echo "Failed to install clang python package automatically. You may need to install it manually into the pixi environment." >&2
    fi
  fi

  # Ensure nanobind is available; CMake skips the Python module when it's missing.
  # nanobind is declared as a dependency in pixi.toml so it should be present, but
  # warn the user if it's unexpectedly missing.
  if ! python3 -c "import nanobind" >/dev/null 2>&1; then
    echo "WARNING: nanobind python package not found. Python bindings will be skipped by CMake." >&2
    echo "Ensure the pixi Python environment includes nanobind (it should be declared in pixi.toml)." >&2
  fi

  # If nanobind is available, compute its CMake config directory and add it
  # to the CMake search path. The conda package installs 'nanobind-config.cmake' under
  # site-packages/nanobind/cmake, which CMake won't find by default.
  if python3 -c "import nanobind" >/dev/null 2>&1; then
    nb_cmake_dir=$(python3 -c "import nanobind, os; print(os.path.join(os.path.dirname(nanobind.__file__), 'cmake'))")
    if [ -d "$nb_cmake_dir" ]; then
      echo "Adding nanobind cmake dir to CMAKE_PREFIX_PATH: $nb_cmake_dir"
      # Build the CMAKE_PREFIX_PATH carefully, avoiding trailing colons
      if [ -z "$CMAKE_PREFIX_PATH" ]; then
        CMAKE_PREFIX_PATH="$nb_cmake_dir"
      else
        CMAKE_PREFIX_PATH="$nb_cmake_dir:$CMAKE_PREFIX_PATH"
      fi
      CMAKE_FLAGS="$CMAKE_FLAGS -DCMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH"
    fi
  fi

  # Try to auto-set LIBCLANG_PATH, preferring the pixi environment's clang
  # if available, since it should provide a matching libclang.so for the
  # clang Python package.
  if [ -z "$LIBCLANG_PATH" ]; then
    # Try pixi environment first
    if [ -n "$CONDA_PREFIX" ]; then
      PIXI_LIBCLANG=$(find "$CONDA_PREFIX/lib" -name "libclang.so*" 2>/dev/null | head -n 1)
      if [ -n "$PIXI_LIBCLANG" ]; then
        export LIBCLANG_PATH="$PIXI_LIBCLANG"
        export LD_LIBRARY_PATH="$(dirname "$PIXI_LIBCLANG"):$LD_LIBRARY_PATH"
        echo "Auto-set LIBCLANG_PATH from pixi environment: $LIBCLANG_PATH"
      fi
    fi
  fi

  # If still not found, fall back to platform-specific system detection
  if [ -z "$LIBCLANG_PATH" ] && [ "$(uname)" = "Darwin" ]; then
    if command -v brew >/dev/null 2>&1; then
      BREW_LLVM_PREFIX=$(brew --prefix llvm 2>/dev/null || true)
      if [ -n "$BREW_LLVM_PREFIX" ] && [ -f "$BREW_LLVM_PREFIX/lib/libclang.dylib" ]; then
        export LIBCLANG_PATH="$BREW_LLVM_PREFIX/lib/libclang.dylib"
        echo "Auto-set LIBCLANG_PATH=$LIBCLANG_PATH"
      fi
    fi
  fi

  # Linux system fallback
  if [ -z "$LIBCLANG_PATH" ] && [ "$(uname)" = "Linux" ]; then
    # Find the library file (accepting .so, .so.1, etc.)
    POTENTIAL_PATH=$(find /usr/lib/llvm-* /usr/lib/x86_64-linux-gnu -name "libclang.so*" 2>/dev/null | head -n 1)

    if [ -n "$POTENTIAL_PATH" ]; then
      export LIBCLANG_PATH="$POTENTIAL_PATH"

      # CRITICAL: Extract the directory and add it to the linker path
      LIB_DIR=$(dirname "$POTENTIAL_PATH")
      export LD_LIBRARY_PATH="$LIB_DIR:$LD_LIBRARY_PATH"

      echo "Auto-set LIBCLANG_PATH=$LIBCLANG_PATH"
      echo "Added $LIB_DIR to LD_LIBRARY_PATH"

      # Persist the variable for later build invocations by creating an
      # activation hook in the conda environment.  This ensures subsequent
      # `pixi run` commands will have LIBCLANG_PATH exported even during the
      # build step when cmake isn’t running the configure script.
      if [ -n "$CONDA_PREFIX" ] && [ -d "$CONDA_PREFIX/etc/conda/activate.d" ]; then
        mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
        echo "export LIBCLANG_PATH=$LIBCLANG_PATH" > "$CONDA_PREFIX/etc/conda/activate.d/pixi-libclang.sh"
      fi

      # If we know the clang binary version, try to pin the pip clang package
      # to the corresponding major version so the Python bindings are
      # compatible with the shared library we found.  Without this, pip will
      # install the latest clang (e.g. 21.x) which may not match older
      # libclang.so files and results in undefined symbol errors at runtime.
      if command -v clang >/dev/null 2>&1; then
        # Extract the first dotted numeric version from the clang output.
        # Examples of clang --version output vary; prefer a regex search.
        raw_ver=$(clang --version | tr -d '\n' || true)
        ver=$(echo "$raw_ver" | grep -oP '\\d+(\\.\\d+)+' | head -n 1 || true)
        if [ -n "$ver" ]; then
          major=$(echo "$ver" | cut -d. -f1)
          if printf '%s' "$major" | grep -qE '^[0-9]+$'; then
            # Only attempt install if python clang is missing or mismatched.
            pyver=$(python3 -c "import sys
try:
    import clang
    print(getattr(clang, '__version__', ''))
except Exception:
    print('')
" 2>/dev/null || true)
            if ! printf '%s' "$pyver" | grep -q "^${major}\\."; then
              echo "Installing clang python package matching libclang ${major}.x"
              if python3 -m pip install "clang<${major}.0,>=${major}.0"; then
                echo "Installed clang python package ${major}.x"
              else
                echo "warning: failed to install pinned clang package" >&2
              fi
            fi
          fi
        fi
      fi
    fi
  fi

  # Install Bun for JavaScript/WASM bindings tests if not already installed
  # Bun's installer typically places it in ~/.bun/bin and does not modify PATH
  # for non-interactive shells (like pixi tasks). Ensure we can find it.
  if [ -z "$BUN_INSTALL" ] && [ -d "$HOME/.bun" ]; then
    export BUN_INSTALL="$HOME/.bun"
  fi
  if [ -n "$BUN_INSTALL" ] && [ -d "$BUN_INSTALL/bin" ]; then
    export PATH="$BUN_INSTALL/bin:$PATH"
  fi

  if ! command -v bun >/dev/null 2>&1; then
    echo "Bun not found - installing for JavaScript bindings tests..."
    if [ "$(uname)" = "Darwin" ] || [ "$(uname)" = "Linux" ]; then
      if curl -fsSL https://bun.sh/install | bash; then
        export BUN_INSTALL="$HOME/.bun"
        export PATH="$BUN_INSTALL/bin:$PATH"
        echo "Installed Bun to $BUN_INSTALL"
      else
        echo "Warning: Failed to install Bun automatically. JavaScript tests may not work." >&2
      fi
    else
      echo "Warning: Bun auto-install not supported on this platform. Please install manually." >&2
    fi
  else
    echo "Bun already installed: $(command -v bun)"
  fi
fi

if echo "$CMAKE_FLAGS" | grep -q "TINYOPT_CONFIG_JS=ON"; then
  emcmake cmake -B build -G Ninja "$CMAKE_FLAGS"
else
  cmake -B build -G Ninja "$CMAKE_FLAGS"
fi