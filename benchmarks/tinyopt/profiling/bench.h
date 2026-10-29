// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <tinyopt/optimizers/options.h>

namespace tinyopt::benchmark {

/// Micro-profiling configuration shared by all one-shot workloads.
struct BenchConfig {
  int runs = 100000;       ///< Number of optimization calls per timed sample.
  int warmup_runs = 1000;  ///< Untimed optimization calls before sampling.
  int samples = 5;         ///< Number of timed samples used for the statistics.
};

/// Parse `--runs N`, `--warmup N`, and `--samples N` from the command line.
inline BenchConfig ParseBenchConfig(int argc, char** argv, BenchConfig config = {}) {
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--runs" && i + 1 < argc) {
      config.runs = std::atoi(argv[++i]);
    } else if (argument == "--warmup" && i + 1 < argc) {
      config.warmup_runs = std::atoi(argv[++i]);
    } else if (argument == "--samples" && i + 1 < argc) {
      config.samples = std::atoi(argv[++i]);
    }
  }
  config.runs = std::max(1, config.runs);
  config.warmup_runs = std::max(0, config.warmup_runs);
  config.samples = std::max(1, config.samples);
  return config;
}

/// Timing statistics for a single workload.
struct BenchResult {
  double min_us = 0;         ///< Fastest sample, in microseconds per run.
  double median_us = 0;      ///< Median sample, in microseconds per run.
  double p95_us = 0;         ///< 95th percentile sample, in microseconds per run.
  double ns_per_iter = 0;    ///< Median time divided by iterations per run.
  double iters_per_run = 0;  ///< Average number of optimizer iterations per run.
};

/// Options used by the micro-profiling workloads.
///
/// Everything that can add per-iteration noise (logging, history, timing, and
/// the final covariance/Hessian export) is disabled so timings only measure
/// the optimizer loop. The optional fields are probed so that this header can
/// also be compiled against older Tinyopt revisions for before/after runs.
template <typename Opt = Options>
inline Opt PinnedOptions() {
  Opt options;
  options.stop.max_iters = 100;
  options.stop.min_error = 1e-12f;
  options.stop.min_rerr_dec = 1e-6f;
  options.stop.min_step_norm2 = 1e-16f;
  options.stop.min_grad_norm2 = 0.0f;
  options.stop.max_consec_failures = 10;
  if constexpr (requires(Opt& candidate) { candidate.stop.min_grad_max; }) {
    options.stop.min_grad_max = 1e-9f;
  }
  options.log.enable = false;
  if constexpr (requires(Opt& candidate) { candidate.exp.final_cov; }) {
    options.exp.final_cov = false;
    options.exp.save_history = false;
    options.exp.measure_time = false;
  } else {
    options.hessian.save_last = false;
    options.save_history = false;
    options.measure_time = false;
  }
  return options;
}

/// Time a workload that can be repeated `config.runs` times per sample.
///
/// `run_once(run)` performs one optimization and returns the number of
/// optimizer iterations it took. The function is called once per warmup and
/// once per timed run, so it must be side-effect free apart from the counters
/// the caller wants to accumulate.
template <typename Fn>
BenchResult TimeWorkload(const char* name, const BenchConfig& config, Fn&& run_once) {
  for (int run = 0; run < config.warmup_runs; ++run) run_once(run);

  std::vector<double> us_per_run(static_cast<std::size_t>(config.samples));
  double iters_per_run = 0;
  for (int sample = 0; sample < config.samples; ++sample) {
    std::size_t iterations = 0;
    const auto start = std::chrono::steady_clock::now();
    for (int run = 0; run < config.runs; ++run) {
      iterations += static_cast<std::size_t>(run_once(run));
    }
    const auto stop = std::chrono::steady_clock::now();
    us_per_run[static_cast<std::size_t>(sample)] =
        std::chrono::duration<double, std::micro>(stop - start).count() / config.runs;
    iters_per_run = static_cast<double>(iterations) / config.runs;
  }

  std::vector<double> sorted = us_per_run;
  std::sort(sorted.begin(), sorted.end());
  const std::size_t count = sorted.size();
  const std::size_t p95_index =
      std::min(count - 1, static_cast<std::size_t>(0.95 * static_cast<double>(count)));

  BenchResult result;
  result.min_us = sorted.front();
  result.median_us = sorted[count / 2];
  result.p95_us = sorted[p95_index];
  result.iters_per_run = iters_per_run;
  result.ns_per_iter = iters_per_run > 0 ? 1e3 * result.median_us / iters_per_run : 0;

  std::printf(
      "BENCH name=%s runs=%d samples=%d iters_per_run=%.3f min_us=%.3f median_us=%.3f "
      "p95_us=%.3f ns_per_iter=%.1f\n",
      name, config.runs, config.samples, result.iters_per_run, result.min_us, result.median_us,
      result.p95_us, result.ns_per_iter);
  std::printf("%-4s median %8.3f us/run  min %8.3f  p95 %8.3f  %8.1f ns/iter\n", name,
              result.median_us, result.min_us, result.p95_us, result.ns_per_iter);
  return result;
}

}  // namespace tinyopt::benchmark
