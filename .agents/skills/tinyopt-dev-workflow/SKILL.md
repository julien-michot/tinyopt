---
name: tinyopt-dev-workflow
description: >-
  Use this skill when developing, building, compiling with -Werror, formatting, or debugging
  tests in the Tinyopt repository. Covers pixi environments, clean builds, ctest execution,
  and clang-format styling.
---

# Tinyopt Development Workflow

This runbook guides you through building, compiling, and testing the Tinyopt C++20 optimization library with zero compiler warnings and strict quality adherence.

---

## 1. Quick Test Run

To build and run all unit tests in a single command, execute the provided helper script:

```shell
./.agents/skills/tinyopt-dev-workflow/scripts/run_tests.sh
```

To run a specific test with verbose Catch2 output:
```shell
./.agents/skills/tinyopt-dev-workflow/scripts/run_tests.sh tinyopt_test_sqrt2
```

To force a clean rebuild before running tests:
```shell
./.agents/skills/tinyopt-dev-workflow/scripts/run_tests.sh --clean
```

---

## 2. Step-by-Step Manual Workflow

### Step 2.1: Clean Previous Artifacts
If switching environments or encountering mysterious build/link errors, clean the build directory:
```shell
rm -rf build
```
See [pixi_environments.md](./references/pixi_environments.md) for details on avoiding CMake cache cross-contamination.

### Step 2.2: Configure in the `test` Pixi Environment
```shell
pixi run -e test bash pixi-configure.sh -DTINYOPT_BUILD_TESTS=ON
```

### Step 2.3: Compile with Ninja
Tinyopt enforces `-Wall -Wextra -Werror`. Ensure zero compiler warnings:
```shell
pixi run -e test cmake --build build
```

### Step 2.4: Execute Tests
```shell
cd build && ctest --output-on-failure
```
Or run a single test target directly:
```shell
./build/tests/tinyopt_test_optimize_easy -s
```

---

## 3. Code Formatting & Linting

All code must strictly follow the repository's `.clang-format` (Google-based, 2 spaces indentation, 100 character column limit).

To format all files in place:
```shell
./.agents/skills/tinyopt-dev-workflow/scripts/format.sh
```

To dry-run and verify if any files violate formatting:
```shell
./.agents/skills/tinyopt-dev-workflow/scripts/format.sh --check
```

---

## 4. Verification Checklist Before Marking Work Complete

1. [ ] Build succeeds with zero warnings under `-Werror`.
2. [ ] All 25+ Catch2 unit tests pass (`ctest --output-on-failure`).
3. [ ] Any new test target has been registered in `tests/CMakeLists.txt` via `add_executable(...)` and `add_test_target(...)`.
4. [ ] Code has been formatted using `./.agents/skills/tinyopt-dev-workflow/scripts/format.sh`.
5. [ ] Header files include the required Apache-2.0 copyright header.
