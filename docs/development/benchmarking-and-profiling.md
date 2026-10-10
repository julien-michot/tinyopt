# Benchmarking and Profiling

## Tinyopt Benchmarks

Run the Tinyopt suite with:

```sh
pixi run bench
```

The comparison suite for Tinyopt, Ceres, Ceres `TinySolver`, g2o, and GTSAM is:

```sh
pixi run bench-all
```

It compares deterministic dense math problems, sparse chain least-squares systems, robust pose
problems, and bundle-adjustment workloads where supported. The dense suite covers static and dynamic
1D, 2D, and 3D parameters, plus dynamic prior cases of sizes 6, 12, 33, and 50. Sparse cases use
10, 100, and 1000 scalar parameters with local unary and pairwise residuals. Bundle adjustment
compares 5/50, 20/200, and 50/500 camera/point problems, with each point observed by a consecutive
subset of cameras.

The runner produces runtime tables, plots, bundle-adjustment iteration counts, stopping criteria,
and machine/build configuration. Tinyopt uses direct accumulation functions where applicable;
third-party implementations provide analytical Jacobians. `bench-all` writes plot images and a styled
HTML report under `build-bench-all/benchmark-report/`. Add `--plot` to display plots and wait for them
to close, `--show` to open the report in a browser, or `--only tinyopt ceres` to select backends.

The g2o Pixi package is used where available; on macOS, CMake fetches the pinned g2o repository.

## Linux `perf` Profiling

The profiling programs are separate from the Catch2 benchmark executables. Build and profile both
workloads with:

```sh
pixi run profilings
```

To profile one workload, pass `1d`, `2d` or `sparse-ba`. The sampling frequency and output folder can also
be changed:

```sh
pixi run profilings 2d
pixi run profilings sparse-ba --frequency 500 --output-dir tmp/profiles
```

The build creates only the dedicated 2D and sparse bundle-adjustment executables under
`benchmarks/tinyopt/profiling/`. Each prints its final cost and parameters, with optimizer logging,
timings, history, and final Hessian saving disabled. The profiler records user-space samples, prints
a top-50 function-overhead table, and writes the raw `.data`, text report, and self-contained HTML
report with an embedded hotspot graph under `tmp/profiles/`.

Open a saved profile with `perf report -i tmp/profiles/<profile>.data` or
`perf report --stdio -i tmp/profiles/<profile>.data`. If perf denies CPU events, set
`kernel.perf_event_paranoid` to `0` or grant the process `CAP_PERFMON`; this requires administrator
access. Kernel symbols may remain unavailable on hosts that restrict `/proc/kallsyms`.

Use tools like `hotspot` to visualize the call graph.
