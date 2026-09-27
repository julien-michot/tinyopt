# Pixi Environments & Build Caching Guide

Tinyopt uses [Pixi](https://pixi.prefix.dev/) to bundle compilers, Eigen, Catch2, Sophus, Ceres Solver, and documentation tools into distinct Conda environments defined in `pixi.toml`.

## Environment Breakdown

| Environment | Included Features | Purpose |
| :--- | :--- | :--- |
| `default` | Base compilers, CMake, Ninja, Eigen | Minimal environment for installing headers or building packages |
| `test` | Base + Catch2 + Sophus | Primary development environment for compiling and executing unit tests |
| `bench` | Base + Catch2 + Ceres Solver | Performance benchmarking and Ceres solver comparisons |
| `docs` | Base + Python + Sphinx + Doxygen + Breathe | Building HTML/RTD documentation |
| `all` | Base + Sophus + Ceres | Comprehensive environment with all third-party integrations |

## Critical Rule: Avoid CMake Cache Cross-Contamination

When CMake configures the project, it saves absolute paths to dependencies (such as Eigen, Catch2, or Ceres) in `build/CMakeCache.txt`.

If you configure using one Pixi environment (e.g. `test`) and subsequently attempt to build or configure using another (e.g. `python` or `bench`), CMake may link against library binaries from one environment while including headers from another, triggering linker errors (such as missing symbol definitions or ABI mismatches).

### Solution: Clean Before Switching Environments
Always wipe `build/` before switching between environments:
```shell
# Option A: Pixi task
pixi run clean

# Option B: Manual cleanup
rm -rf build
```

## Running Tasks Within Specific Environments

Always specify the environment flag `-e <env>` when executing tasks if the task relies on environment-specific dependencies:

```shell
# Configure and run tests
pixi run tests

# Run benchmarks
pixi run bench
```
