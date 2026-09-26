## Bindings — C and Python

This folder contains code and helpers for the bindings shipped with tinyopt.

- C API bindings are in `bindings/c/` (C API header `api.h`, narrow C++ wrappers, and a CMake target).
- Python bindings are implemented using nanobind and live under `bindings/python/` plus the generator script `bindings/gen_struct_bindings.py`.

### Contract (quick)
- Inputs: repository source, optional `libclang` and `nanobind` development files.
- Outputs: `tinyopt` C API library, and optionally a `tinyopt` Python extension when nanobind is available.
- Success: C bindings always built when `TINYOPT_BUILD_BINDINGS=ON`; Python extension built only if `nanobind` is found.

### Requirements
- CMake (>= 3.25), a C++ toolchain, and a working Python 3 dev environment if you want Python bindings.
- For the generator: Python 3.9+, `numpy`, and the `clang` Python package (provides `clang.cindex`) so the generator can parse headers via libclang.
- You also need a libclang shared library installed (commonly provided by an LLVM/Clang runtime package).

Example macOS hints
- Install LLVM via Homebrew and export the `LIBCLANG_PATH` if needed:

```bash
brew install llvm
export LIBCLANG_PATH=/opt/homebrew/opt/llvm/lib/libclang.dylib
pip install clang numpy
```

### Notes about CMake behavior
The project's `cmake/Bindings.cmake` requires Python 3 development components and will try to find `nanobind` quietly. In short:

- `find_package(Python 3 COMPONENTS Interpreter Development REQUIRED)` — Python dev components are required for the bindings build path.
- `find_package(nanobind QUIET)` — nanobind is optional. If `nanobind` is not found, the build will skip the Python extension and print a message like "nanobind not found - skipping Python bindings".

This means the C bindings target can be configured and built even if nanobind isn't available; the Python extension is only built when nanobind is present.

### Generating struct bindings
The small generator parses headers via libclang and writes a C++ file that is included in the Python extension build.

Example (from repo root):

```bash
python3 bindings/gen_struct_bindings.py \
  include/tinyopt/optimizers/options.h \
  include/tinyopt/output.h \
  build/bindings/python/generated_struct_bindings.cpp \
  include
```

The generator requires `clang.cindex` (the `clang` Python package) and access to a libclang shared library. If it fails with a missing `clang.cindex` error, install the Python `clang` package and make sure `LIBCLANG_PATH` points at your libclang shared library.

### Using pixi (recommended)
The project provides `pixi` tasks in `pixi.toml` to configure, build and test bindings. The relevant tasks are:

- `configure-bindings` — configures the build with `-DTINYOPT_BUILD_BINDINGS=ON -DTINYOPT_BUILD_TESTS=ON` and uses the `python` pixi environment (installs `nanobind`, `clang`, `numpy` when available).
- `build-bindings` — runs the build after configuration.
- `test-bindings` — runs binding tests (CTEST label: `Bindings`).

Run the sequence from the repo root (pixi must be installed):

```bash
pixi run configure-bindings
pixi run build-bindings
pixi run test-bindings
```

These tasks map to the `pixi.toml` entries and are kept up-to-date there. If you prefer to run CMake manually, see below.
Bindings README (clean)

This is a short placeholder to confirm the file has been overwritten. I'll expand it to the full README next.

- Outputs: `tinyopt` C API library, and optionally a `tinyopt` Python extension when nanobind is available.
- Success: C bindings always built when `TINYOPT_BUILD_BINDINGS=ON`; Python extension built only if `nanobind` is found.

### Requirements
- CMake (>= 3.25), a C++ toolchain, and a working Python 3 dev environment if you want Python bindings.
- For the generator: Python 3.9+, `numpy`, and the `clang` Python package (provides `clang.cindex`) so the generator can parse headers via libclang.
- You also need a libclang shared library installed (commonly provided by an LLVM/Clang runtime package).

Example macOS hints
- Install LLVM via Homebrew and export the `LIBCLANG_PATH` if needed:

```bash
brew install llvm
export LIBCLANG_PATH=/opt/homebrew/opt/llvm/lib/libclang.dylib
pip install clang numpy
```

### Notes about CMake behavior
The project's `cmake/Bindings.cmake` requires Python 3 development components and will try to find `nanobind` quietly. In short:

- `find_package(Python 3 COMPONENTS Interpreter Development REQUIRED)` — Python dev components are required for the bindings build path.
- `find_package(nanobind QUIET)` — nanobind is optional. If `nanobind` is not found, the build will skip the Python extension and print a message like "nanobind not found - skipping Python bindings".

This means the C bindings target can be configured and built even if nanobind isn't available; the Python extension is only built when nanobind is present.

### Generating struct bindings
The small generator parses headers via libclang and writes a C++ file that is included in the Python extension build.

Example (from repo root):

```bash
python3 bindings/gen_struct_bindings.py \
  include/tinyopt/optimizers/options.h \
  include/tinyopt/output.h \
  build/bindings/python/generated_struct_bindings.cpp \
  include
```

The generator requires `clang.cindex` (the `clang` Python package) and access to a libclang shared library. If it fails with a missing `clang.cindex` error, install the Python `clang` package and make sure `LIBCLANG_PATH` points at your libclang shared library.

### Using pixi (recommended)
The project provides `pixi` tasks in `pixi.toml` to configure, build and test bindings. The relevant tasks are:

- `configure-bindings` — configures the build with `-DTINYOPT_BUILD_BINDINGS=ON -DTINYOPT_BUILD_TESTS=ON` and uses the `python` pixi environment (installs `nanobind`, `clang`, `numpy` when available).
- `build-bindings` — runs the build after configuration.
- `test-bindings` — runs binding tests (CTEST label: `Bindings`).

Run the sequence from the repo root (pixi must be installed):

```bash
pixi run configure-bindings
pixi run build-bindings
pixi run test-bindings
```

These tasks map to the `pixi.toml` entries and are kept up-to-date there. If you prefer to run CMake manually, see below.
## Bindings — C and Python

This folder contains code and helpers for the bindings shipped with tinyopt.

- C API bindings are in `bindings/c/` (C API header `api.h`, narrow C++ wrappers, and a CMake target).
- Python bindings are implemented using nanobind and live under `bindings/python/` plus the generator script `bindings/gen_struct_bindings.py`.

### Contract (quick)
- Inputs: repository source, optional `libclang` and `nanobind` development files.
- Outputs: `tinyopt` C API library, and optionally a `tinyopt` Python extension when nanobind is available.
- Success: C bindings always built when `TINYOPT_BUILD_BINDINGS=ON`; Python extension built only if `nanobind` is found.

### Requirements
- CMake (>= 3.25), a C++ toolchain, and a working Python 3 dev environment if you want Python bindings.
- For the generator: Python 3.9+, `numpy`, and the `clang` Python package (provides `clang.cindex`) so the generator can parse headers via libclang.
- You also need a libclang shared library installed (commonly provided by an LLVM/Clang runtime package).

Example macOS hints
- Install LLVM via Homebrew and export the `LIBCLANG_PATH` if needed:

```bash
brew install llvm
export LIBCLANG_PATH=/opt/homebrew/opt/llvm/lib/libclang.dylib
pip install clang numpy
```

### Notes about CMake behavior
The project's `cmake/Bindings.cmake` requires Python 3 development components and will try to find `nanobind` quietly. In short:

- `find_package(Python 3 COMPONENTS Interpreter Development REQUIRED)` — Python dev components are required for the bindings build path.
- `find_package(nanobind QUIET)` — nanobind is optional. If `nanobind` is not found, the build will skip the Python extension and print a message like "nanobind not found - skipping Python bindings".

This means the C bindings target can be configured and built even if nanobind isn't available; the Python extension is only built when nanobind is present.

### Generating struct bindings
The small generator parses headers via libclang and writes a C++ file that is included in the Python extension build.

Example (from repo root):

```bash
python3 bindings/gen_struct_bindings.py \
  include/tinyopt/optimizers/options.h \
  include/tinyopt/output.h \
  build/bindings/python/generated_struct_bindings.cpp \
  include
```

The generator requires `clang.cindex` (the `clang` Python package) and access to a libclang shared library. If it fails with a missing `clang.cindex` error, install the Python `clang` package and make sure `LIBCLANG_PATH` points at your libclang shared library.

### Using pixi (recommended)
The project provides `pixi` tasks in `pixi.toml` to configure, build and test bindings. The relevant tasks are:

- `configure-bindings` — configures the build with `-DTINYOPT_BUILD_BINDINGS=ON -DTINYOPT_BUILD_TESTS=ON` and uses the `python` pixi environment (installs `nanobind`, `clang`, `numpy` when available).
- `build-bindings` — runs the build after configuration.
- `test-bindings` — runs binding tests (CTEST label: `Bindings`).

Run the sequence from the repo root (pixi must be installed):

```bash
pixi run configure-bindings
pixi run build-bindings
pixi run test-bindings
```

These tasks map to the `pixi.toml` entries and are kept up-to-date there. If you prefer to run CMake manually, see below.

### Manual CMake build
To configure and build only bindings using CMake directly from the repo root:

```bash
# configure (enable bindings)
cmake -S . -B build -DTINYOPT_BUILD_BINDINGS=ON -DTINYOPT_BUILD_TESTS=ON
# build C bindings (and Python extension if nanobind is found)
cmake --build build --target tinyopt_bindings -j$(sysctl -n hw.ncpu)
# optionally run binding tests
cd build && ctest -L Bindings -V
```

Replace `tinyopt_bindings` with the target name used in your build if it differs; inspect the generated `build.ninja` or `build/` CMake files for exact target names on your platform.

### C bindings
The C API lives under `bindings/c/` and exposes a C-friendly surface (`bindings/c/api.h`) so consumers who don't use C++ can link against tinyopt. Build and packaging of the C API are handled by the main CMake build when `TINYOPT_BUILD_BINDINGS=ON`.

### Troubleshooting
- If the Python extension doesn't build: check that `nanobind` and Python development files are installed. The CMake configure step prints whether nanobind was found.
- If the generator fails due to `libclang` missing: install an LLVM/Clang runtime and set `LIBCLANG_PATH` to the shared library location.
- On macOS you may encounter linking differences depending on which Python you use (system/framework vs Homebrew). If you see unresolved Python C-API symbols at link time, try adding `-undefined dynamic_lookup` to the extension link flags locally or link explicitly against your Python framework.

### Next steps and small improvements
- Add a tiny CI job to verify that the Python bindings build when `nanobind` is available.
- Add a short `bindings/README_quickstart.md` example showing how to import and use the Python module once built.

If you'd like, I can add a small helper that auto-detects `libclang` and suggests a `LIBCLANG_PATH` value on failure.