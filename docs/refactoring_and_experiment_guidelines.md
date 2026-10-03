# Refactoring and Experiment Guidelines

This document is the required workflow for refactorings and code experiments in Tinyopt. It applies
across the repository and preserves behavioral correctness and measurable performance evidence.

## Before Starting

1. Read the relevant architecture, API, and test documentation before choosing an implementation.
2. State the intended behavior, invariants, and the smallest check that could disprove the approach.
3. For work under `include/` (and `src/` if introduced), capture a baseline for the current `HEAD`
   before editing:

   ```shell
   pixi run refactor-timings
   pixi run refactor-timings --finalize
   ```

   The first command stores current measurements as `timings/dev.json`; the second pins them to
   `timings/<HEAD-short-hash>.json` for comparisons during the experiment.
4. Keep each experiment focused. Record the hypothesis and avoid mixing unrelated behavior changes.

## Implementation and Validation

1. Preserve public behavior unless an API change is explicitly part of the goal. Document any
   changed contract and add focused regression coverage.
2. Run relevant tests and correctness checks while iterating. Validate derivative changes against
   numerical differentiation and optimizer changes for convergence where applicable.
3. Do not use benchmark results as a substitute for correctness tests.
4. After all code edits and other checks are complete, run this as the final validation step for
   changes under `include/` or `src/`:

   ```shell
   pixi run refactor-timings
   ```

   This clean-compiles the test targets, builds the non-Ceres benchmark executable, runs the
   benchmarks three times, and compares both measurements against the timing file for the current
   `HEAD` commit. Do not edit code after this final measurement without running it again.
5. Review both comparison rows. Investigate material regressions before accepting the change; a
   missing baseline is not evidence of parity. The script labels changes within 5% as neutral to
   avoid overstating timing noise.
6. After committing, associate the final `dev` measurements with the new commit:

   ```shell
   pixi run refactor-timings --finalize
   ```

   This renames `timings/dev.json` to the new `timings/<HEAD-short-hash>.json`. Timing files are
   intentionally local and ignored by Git.

## Measurement Details

- Optimize compilation time is the wall time for building the dedicated
   `tinyopt_timing_optimize` target after cleaning `build-tests`; that minimal test calls
   `Optimize()` on both `Vec2` and `VecXf`. Other test targets are excluded, and CMake
   configuration is not timed.
- The benchmark metric is the median, across three XML reports, of the sum of Catch2's per-case
   mean durations from the non-Ceres benchmark suite. Catch2 reports these means in nanoseconds;
   reports are retained in the ignored `timings/` directory. The executable is rebuilt before
   measurement.
- Comparisons always use the timing file named for the current `HEAD` short hash. If that file is
  missing, capture and finalize a baseline before proceeding with the experiment.
- Record the displayed before/after values and status in the change summary. Report failures or
  missing results plainly; do not claim a speedup or parity without measurements.
- Use `pixi run plot-refactor-timings` to generate `timings/evolution.png` from finalized result
   files. Plot points use each commit's recorded timestamp and short hash.

## Results Table

The timing task prints a table with both required metrics:

| Metric | Before (`HEAD`) | Current (`dev`) | Delta | Status |
| --- | ---: | ---: | ---: | --- |
| Minimal Optimize() test compilation | seconds | seconds | seconds and percent | faster / slower / neutral |
| Non-Ceres benchmark mean sum | seconds | seconds | seconds and percent | faster / slower / neutral |
