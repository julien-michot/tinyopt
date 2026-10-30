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

To profile one workload, pass `1d`, `2d`, `6d`, `block-sparse-ba` or `sparse-ba`. The sampling
frequency and output folder can also be changed:

```sh
pixi run profilings 2d
pixi run profilings block-sparse-ba --frequency 500 --output-dir tmp/profiles
pixi run profilings sparse-ba --frequency 500 --output-dir tmp/profiles
```

The build creates dedicated micro and bundle-adjustment executables under
`benchmarks/tinyopt/profiling/`. The `1d`, `2d`, and `6d` micro workloads reuse one optimizer across
runs so they measure the optimizer loop rather than optimizer construction. `2d` and `6d` cover the
closed-form small-system solver and the dense Eigen solver path respectively. Each workload prints a
machine-readable `BENCH name=... median_us=... ns_per_iter=...` line; the micro workloads also accept
`--runs`, `--warmup`, and `--samples` to trade runtime for stability. Optimizer logging, timings,
history, and final covariance export are disabled in all profiling workloads.

The profiler records user-space samples, prints a top-50 function-overhead table, and writes the raw
`.data`, text report, and self-contained HTML report with an embedded hotspot graph under
`tmp/profiles/`.

### Before/after comparison

Use the comparison task to build the *same* profiling sources against two revisions and report the
per-run time ratio. It isolates library changes from benchmark-options drift:

```sh
pixi run profiling-compare 955242b 35815ef

# Advanced flags are available by calling the script directly:
python benchmarks/scripts/profiling_compare.py --before 955242b --after 35815ef \
    --runs 500000 --samples 7 --workload 2d --workload 6d
```

The script prints one row per workload (`1d`, `2d`, `6d`) and flags changes above 5% as slower. It
compares committed revisions; commit or stash working-tree changes first if you want them included.

Open a saved profile with `perf report -i tmp/profiles/<profile>.data` or
`perf report --stdio -i tmp/profiles/<profile>.data`. If perf denies CPU events, set
`kernel.perf_event_paranoid` to `0` or grant the process `CAP_PERFMON`; this requires administrator
access. Kernel symbols may remain unavailable on hosts that restrict `/proc/kallsyms`.

Use tools like `hotspot` to visualize the call graph.

---

## Solver Configuration & Stopping Criteria Mapping

To ensure reproducible and apples-to-apples comparisons across Tinyopt, Ceres Solver, g2o, and GTSAM, benchmark configurations must align on both outer stopping criteria and inner Levenberg-Marquardt (LM) settings.

### Comparison Matrix

| Concept | Tinyopt (`tinyopt::Options`) | Ceres (`ceres::Solver::Options`) | g2o (`G2oTerminationAction`) | GTSAM (`gtsam::LevenbergMarquardtParams`) | Definition & Metric Differences |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Relative Cost Decrease** | `stop.min_rerr_dec`<br>(default: `1e-10`, bench: `1e-6`) | `function_tolerance`<br>(default: `1e-6`, bench: `1e-6`) | `relative_decrease`<br>(bench: `1e-6`) | `relativeErrorTol`<br>(default: `1e-5`, bench: `1e-6`) | Relative change in objective: $\frac{f_{k-1} - f_k}{f_{k-1}} < \text{tol}$. All backends evaluate this directly. |
| **Absolute Cost Floor** | `stop.min_error`<br>(default: `1e-12`, bench: `1e-12`) | *(None / floating zero)* | *(None)* | `errorTol`<br>(bench: `1e-12`) | Tinyopt terminates if $f_k < \text{min\_error}$. Ceres does not have an absolute cost floor by default, so on zero-residual problems Ceres runs until parameter or relative tolerance. |
| **Gradient Tolerance** | `stop.min_grad_norm2` (squared $\ell_2$) or `stop.min_grad_max` ($\ell_\infty$ norm)<br>(default: `1e-18` / `0.0`, bench: `min_grad_max = 1e-9`) | `gradient_tolerance`<br>(default: `1e-10`, bench: `1e-9`) | *(Not evaluated in action)* | *(None / implicit)* | Ceres checks **$\ell_\infty$-norm** ($\|\nabla f\|_\infty \le \text{tol}$, max coordinate). Tinyopt supports both squared $\ell_2$ (`min_grad_norm2`) and $\ell_\infty$ (`min_grad_max`). Using `min_grad_max = 1e-9` exactly matches Ceres `gradient_tolerance`. |
| **Parameter Step Tolerance** | `stop.min_step_norm2` (squared step) or `stop.min_step_rel_norm2` (relative)<br>(default: `1e-14` / `0.0`, bench: `1e-16`) | `parameter_tolerance`<br>(default: `1e-8`, bench: `1e-8`) | `step_norm_squared < 1e-16` | `absoluteErrorTol = 0` | **Metric difference:** Tinyopt checks **squared step norm** ($\|\Delta x\|_2^2 < \text{tol}$) or relative squared step norm via `min_step_rel_norm2` ($\frac{\|\Delta x\|_2^2}{\|x\|_2^2 + \text{tol}} < \text{tol}$, matching Ceres). Ceres checks **relative step norm** ($\frac{\|\Delta x\|_2}{\|x\|_2 + \text{tol}} < \text{tol}$). |
| **Max Outer Iterations** | `stop.max_iters`<br>(default: `50`, bench: `100`) | `max_num_iterations`<br>(default: `50`, bench: `100`) | `max_iterations`<br>(bench: `100`) | `maxIterations`<br>(bench: `100`) | Maximum successful iterations (or total iterations depending on failure handling). |
| **Max Consecutive Failures** | `stop.max_consec_failures`<br>(default: `5`, bench: `10`) | `max_num_consecutive_invalid_steps`<br>(default: `5`, bench: `3`) | `setMaxTrialsAfterFailure`<br>(bench: `3`) | `maxConsecutiveInvalidSteps` | Number of consecutive rejected damping increases allowed before aborting. |
| **LM Initial Damping** | `lm.damping_init`<br>(default: `1e-4`, bench: `1e-4`) | `initial_trust_region_radius`<br>(bench: `1e4` $\implies \lambda = 10^{-4}$) | `setUserLambdaInit`<br>(bench: `1e-4`) | `lambdaInitial`<br>(bench: `1e-4`) | Initial damping $\lambda_0$. In Ceres LM, initial damping relates to trust region radius $\mu = 1 / \Delta_0 = 10^{-4}$. |
| **LM Jacobi / Diagonal Scaling** | `lm.jacobi_scaling`<br>(default: `false`, bench: `true`) | `jacobi_scaling`<br>(default: `true`, bench: `true`) | Enabled in block solver | Implicit in scaling | Scales system by $D = \text{diag}(H)^{1/2}$ before damping $(H + \lambda D^T D)$. |
| **Step Quality Metric** | Actual vs. predicted quadratic model decrease $\rho = \frac{f(x) - f(x+\Delta x)}{L(0) - L(\Delta x)}$ (exact Nielsen update) | Actual vs. predicted reduction $\rho = \frac{f(x) - f(x+\Delta x)}{L(0) - L(\Delta x)}$ | Standard Marquardt ratio $\rho = \frac{\Delta f}{\Delta L}$ | Model fidelity $\rho = \frac{\Delta f}{\Delta L}$ | Tinyopt LM evaluates exact predicted model reduction $\Delta L = \frac{1}{2} (-g^T \Delta x + \lambda \Delta x^T D^T D \Delta x)$ without heap allocation and adapts damping dynamically via Nielsen's rule $\lambda \leftarrow \lambda \cdot \max(1/3, 1 - (2\rho - 1)^3)$, matching Ceres and g2o. |
| **Manifold vs. Parameterization** | Direct $\mathbb{R}^6$ (Euler angles + translation) | Direct $\mathbb{R}^6$ (Euler angles + translation) | Lie group $SE(3)$ (`VertexSE3Expmap`) | Lie group `Pose3` / manifold | **Manifold advantage:** g2o optimizes on $SE(3)$ via Lie algebra $\mathfrak{se}(3)$ perturbations ($\exp(\hat{\xi}) T$), eliminating Euler angle trigonometric coupling and singularities. This enables full Newton steps and convergence in $\sim 9$ iterations. |

---

## Rules for Fair Benchmark Comparisons

When adding or updating benchmarks comparing Tinyopt against external solvers (Ceres, g2o, GTSAM):

1. **Use Synchronized Option Factories**:
   All Tinyopt benchmarks must configure their options via `tinyopt::benchmark::CreateOptions()` in `benchmarks/common/options.h` rather than ad-hoc inline initialization.
2. **Harmonize Stopping Metric Types**:
   - Do not compare an absolute step norm directly to a relative parameter tolerance.
   - For high-dimensional problems ($D > 1000$), avoid comparing $\ell_2$ norm to $\ell_\infty$ norm for gradient tolerances without accounting for $\sqrt{D}$ scaling.
   - Set equivalent relative cost decrease thresholds (`min_rerr_dec == function_tolerance == 1e-6`).
3. **Harmonize Cost Floors (`min_error`)**:
   On noise-free synthetic problems (zero ground-truth residual), an absolute cost floor such as `min_error = 1e-12` causes Tinyopt to keep optimizing until floating-point zero even after relative tolerances have satisfied typical termination. If comparing iteration counts to solvers that terminate on relative change, ensure the cost floors are aligned.
4. **Equalize Damping and Scaling Defaults**:
   Ensure `lm.jacobi_scaling` is enabled whenever third-party solvers use diagonal Jacobi preconditioning (e.g., Ceres's default). Match initial damping ($\lambda = 10^{-4}$).
5. **State Parameterization Differences Explicitly**:
   If a benchmark uses Lie algebra / manifold perturbations (such as g2o $SE(3)$) while another uses Euler-angle parameter vectors, explicitly note the parameterization disparity in the benchmark header and report. Manifold updates inherently reduce iteration count by eliminating artificial parameter coupling.

