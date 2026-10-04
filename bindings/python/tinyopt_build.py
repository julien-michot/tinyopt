# Copyright 2026 Julien Michot.
# SPDX-License-Identifier: Apache-2.0
"""Setuptools commands (wired in pyproject.toml) that bundle `libtinyopt_c` built with CMake."""

import os
import shutil
import subprocess
from pathlib import Path

from setuptools.build_meta import *  # noqa: F401,F403  (PEP 517 hooks)
from setuptools.command.build_ext import build_ext

try:
    from setuptools.command.bdist_wheel import bdist_wheel
except ImportError:  # setuptools < 70.1
    from wheel.bdist_wheel import bdist_wheel

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = ROOT / "bindings" / "python" / "tinyopt"
LIB_SUFFIXES = (".so", ".dylib", ".dll")
BUILD_DIR = ROOT / "build" / "cmake-c-library"
(ROOT / "build" / "pip").mkdir(parents=True, exist_ok=True)


class BuildCLibrary(build_ext):
    """Bundles libtinyopt_c: a prebuilt one (TINYOPT_C_LIBRARY) or an incremental CMake build."""

    lib = None

    def _destination(self):
        if self.inplace:  # editable installs load the library from the source tree
            return PACKAGE_DIR
        return Path(self.build_lib) / "tinyopt"

    def _build(self):
        cmake = shutil.which("cmake")
        if cmake is None:
            raise RuntimeError("CMake >= 3.25 is required to build Tinyopt's C library")
        build_dir = BUILD_DIR  # persistent, so repeated installs only rebuild what changed
        build_dir.mkdir(parents=True, exist_ok=True)
        config = [
            cmake, "-S", str(ROOT), "-B", str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DTINYOPT_BUILD_C_LIBRARY=ON", "-DTINYOPT_BUILD_SHARED_C=ON",
            "-DTINYOPT_BUILD_TESTS=OFF", "-DTINYOPT_BUILD_EXAMPLES=OFF",
            "-DTINYOPT_BUILD_BENCHMARKS=OFF", "-DTINYOPT_BUILD_DOCS=OFF",
            "-DTINYOPT_ENABLE_LINEAR_SOLVER_ALL=ON", "-DTINYOPT_ENABLE_OPTIMIZERS_ALL=ON",
            "-DTINYOPT_ENABLE_SUITESPARSE=OFF",
            "-DTINYOPT_WERROR=OFF",  # the user's compiler may warn where ours does not
        ]
        if shutil.which("ninja"):
            config += ["-G", "Ninja"]
        config += os.environ.get("TINYOPT_CMAKE_ARGS", "").split()
        subprocess.check_call(config)
        subprocess.check_call([cmake, "--build", str(build_dir), "--target", "tinyopt_c",
                               "--config", "Release", "--parallel"])
        libs = [p for p in build_dir.rglob("*tinyopt_c*")
                if p.suffix in LIB_SUFFIXES and p.is_file()]
        if not libs:
            raise RuntimeError(f"tinyopt_c shared library not found in {build_dir}")
        return libs[0]

    def run(self):
        prebuilt = os.environ.get("TINYOPT_C_LIBRARY")
        if prebuilt:  # link to an existing library: no CMake, no compiler needed
            self.lib = Path(prebuilt)
            if not self.lib.is_file():
                raise RuntimeError(f"TINYOPT_C_LIBRARY={prebuilt} does not exist")
        else:
            self.lib = self._build()
        dest = self._destination()
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.lib, dest / self.lib.name)

    def get_outputs(self):
        return [str(self._destination() / self.lib.name)] if self.lib else []


class PlatformWheel(bdist_wheel):
    """Platform wheel independent of the Python ABI (the C library is loaded by ctypes)."""

    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        return "py3", "none", super().get_tag()[2]

