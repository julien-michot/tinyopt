// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

// Drives the Tinyopt C API (tinyopt_optimize) compiled to WebAssembly, without any glue code:
// the C structs are written into the module memory using their wasm32 layout.

// `id` is the `tinyopt_solver_t` value of the C API.
export const SOLVERS = [
  { id: 0, name: 'Levenberg-Marquardt', color: '#e6194b' },
  { id: 1, name: 'Gauss-Newton', color: '#3cb44b' },
  { id: 2, name: 'Gradient Descent', color: '#4363d8' },
  { id: 3, name: 'Conjugate Gradient', color: '#f58231' },
  { id: 4, name: 'Dogleg', color: '#911eb4' },
  { id: 5, name: 'BFGS', color: '#00bcd4' },
  { id: 6, name: 'L-BFGS', color: '#ffd700' },
];

// wasm32 byte offsets of the fields used in include/tinyopt/c/c_api_common.h and c_api_double.h.
// test.mjs checks the options ones against the C defaults.
export const LAYOUT = {
  options: { size: 1024, solverType: 0, maxIters: 52, maxConsecutiveFailures: 73,
             checkMinHessianDiagonal: 32, stepCallback: 104, stepCallbackData: 108,
             logEnabled: 112, lmDampingInit: 156, gdLearningRate: 176,
             cgStepSize: 180, doglegRadiusInit: 188, bfgsStepSize: 204,
             lbfgsStepSize: 224 },
  // `size` above is a generous upper bound of sizeof(tinyopt_options_t)
  params: { x: 0, dims: 4, plusEq: 8, size: 12 },
  problem: { type: 0, fn: 4, numResiduals: 8, userData: 12, size: 16 },
  summary: { stopReason: 0, numIters: 4, finalCost: 16, size: 32 },
};
const EVAL_RESIDUALS = 1; // tinyopt_eval_type_t
const EVAL_GRADIENT = 2;
// Dogleg, LM and GN take residuals and a Jacobian; the other solvers only use cost and gradient.
const usesResiduals = (solver) => solver <= 1 || solver === 4;
const SOLVER_PARAM_KEYS = {
  0: 'lm_damping_init',
  1: 'check_min_hessian_diagonal',
  2: 'gd_learning_rate',
  3: 'cg_step_size',
  4: 'dogleg_radius_init',
  5: 'bfgs_step_size',
  6: 'lbfgs_step_size',
};
const STATUS_OK = 0;
const STATUS_OPTIMIZATION_FAILED = 3; // Also reported when max_iters is reached.

export class TinyoptWasmOptions {
  constructor(overrides = {}) {
    Object.assign(this, {
      solverType: 0,
      maxIters: 100,
      maxConsecutiveFailures: 40,
      logEnabled: false,
    }, overrides);
  }
}

export class TinyoptWasmOptimizer {
  constructor(module) {
    this.module = module;
    const rawDefaults = this.defaultOptions();
    this.options = {
      solverType: 0,
      maxIters: rawDefaults.maxIters,
      maxConsecutiveFailures: rawDefaults.maxConsecutiveFailures,
      logEnabled: rawDefaults.logEnabled !== 0,
    };
  }

  /** Returns the legacy raw C defaults expected by the existing wasm tests. */
  defaultOptions() {
    const L = LAYOUT;
    const options = this.module._malloc(L.options.size);
    this.module._tinyopt_options_default(options);
    const result = {
      maxIters: this.module.HEAPU16[(options + L.options.maxIters) >> 1],
      maxConsecutiveFailures: this.module.HEAPU8[options + L.options.maxConsecutiveFailures],
      logEnabled: this.module.HEAP32[(options + L.options.logEnabled) >> 2],
    };
    this.module._free(options);
    return result;
  }

  withOptions(overrides = {}) {
    const next = { ...this.options, ...overrides };
    return new TinyoptWasmOptions({
      ...next,
      logEnabled: next.logEnabled ? 1 : 0,
    });
  }

  /**
   * Minimizes 0.5 * sum(r_i^2) of `problem` from a starting point. The JS wrapper exposes a
   * simple options object while the low-level ABI still uses the raw wasm32 layout internally.
   */
  minimize(problem, start, options = {}) {
    const config = this.withOptions(options);
    const solver = config.solverType;
    const [x0, y0] = Array.isArray(start) ? start : [start.x, start.y];
    const key = SOLVER_PARAM_KEYS[solver];
    const solverParamValue = key !== undefined ? config[key] : undefined;
    return this._minimizeRaw(solver, problem, x0, y0, config.maxIters,
                             config.maxConsecutiveFailures, config.logEnabled, solverParamValue);
  }

  _minimizeRaw(solver, problem, x0, y0, maxIters, maxConsecutiveFailures = 40,
              logEnabled = 0, solverParamValue = undefined) {
    const m = this.module;
    const L = LAYOUT;
    const solverParamKey = SOLVER_PARAM_KEYS[solver];
    const alloc = (size) => m._malloc(size);
    const effectiveMaxConsecutiveFailures = Math.max(maxConsecutiveFailures, 40);
    const residuals = new Float64Array(problem.numResiduals);
    const jacobian = new Float64Array(problem.numResiduals * 2);
    const trajectory = [[x0, y0]];
    let px = x0;
    let py = y0;

    const residualsFn = m.addFunction((xPtr, dims, rPtr, jacPtrPtr) => {
      const x = m.HEAPF64[xPtr >> 3];
      const y = m.HEAPF64[(xPtr >> 3) + 1];
      problem.residuals(x, y, residuals);
      m.HEAPF64.set(residuals, rPtr >> 3);
      const jacPtr = m.HEAP32[jacPtrPtr >> 2];
      if (jacPtr !== 0) {
        problem.jacobian(x, y, jacobian);
        m.HEAPF64.set(jacobian, jacPtr >> 3);
      }
      return 0;
    }, 'iiiiiii');

    const gradientFn = m.addFunction((xPtr, dims, costPtr, gradPtr) => {
      const x = m.HEAPF64[xPtr >> 3];
      const y = m.HEAPF64[(xPtr >> 3) + 1];
      problem.residuals(x, y, residuals);
      let c = 0;
      for (let i = 0; i < problem.numResiduals; ++i) c += residuals[i] * residuals[i];
      m.HEAPF64[costPtr >> 3] = 0.5 * c;
      if (gradPtr !== 0) {
        problem.jacobian(x, y, jacobian);
        let gx = 0;
        let gy = 0;
        for (let i = 0; i < problem.numResiduals; ++i) {
          gx += jacobian[2 * i] * residuals[i];
          gy += jacobian[2 * i + 1] * residuals[i];
        }
        m.HEAPF64[gradPtr >> 3] = gx;
        m.HEAPF64[(gradPtr >> 3) + 1] = gy;
      }
      return 0;
    }, 'iiiiii');

    const stepFn = m.addFunction((dxPtr, dims, isRollback) => {
      px += m.HEAPF32[dxPtr >> 2];
      py += m.HEAPF32[(dxPtr >> 2) + 1];
      if (isRollback) trajectory.pop();
      else trajectory.push([px, py]);
      return 0;
    }, 'iiiii');

    const x = alloc(16);
    const params = alloc(L.params.size);
    const prob = alloc(L.problem.size);
    const options = alloc(L.options.size);
    const summary = alloc(L.summary.size);
    try {
      m.HEAPF64[x >> 3] = x0;
      m.HEAPF64[(x >> 3) + 1] = y0;
      m.HEAP32[(params + L.params.x) >> 2] = x;
      m.HEAP32[(params + L.params.dims) >> 2] = 2;
      m.HEAP32[(params + L.params.plusEq) >> 2] = 0;
      m.HEAP32[(prob + L.problem.type) >> 2] = usesResiduals(solver) ? EVAL_RESIDUALS : EVAL_GRADIENT;
      m.HEAP32[(prob + L.problem.fn) >> 2] = usesResiduals(solver) ? residualsFn : gradientFn;
      m.HEAP32[(prob + L.problem.numResiduals) >> 2] = problem.numResiduals;
      m.HEAP32[(prob + L.problem.userData) >> 2] = 0;
      m._tinyopt_options_default(options);
      m.HEAP32[(options + L.options.solverType) >> 2] = solver;
      m.HEAPU16[(options + L.options.maxIters) >> 1] = maxIters;
      m.HEAPU8[options + L.options.maxConsecutiveFailures] = effectiveMaxConsecutiveFailures;
      m.HEAP32[(options + L.options.logEnabled) >> 2] = logEnabled ? 1 : 0;
      if (solverParamKey !== undefined && Number.isFinite(solverParamValue)) {
        const offset = L.options[solverParamKey.replace(/_([a-z])/g, (_, c) => c.toUpperCase())];
        if (offset !== undefined) m.HEAPF32[(options + offset) >> 2] = solverParamValue;
      }
      m.HEAP32[(options + L.options.stepCallback) >> 2] = stepFn;
      m.HEAP32[(options + L.options.stepCallbackData) >> 2] = 0;

      const status = m._tinyopt_optimize(params, prob, options, summary);
      return {
        status,
        ok: status === STATUS_OK || status === STATUS_OPTIMIZATION_FAILED,
        cost: m.HEAPF64[(summary + L.summary.finalCost) >> 3],
        iters: m.HEAP32[(summary + L.summary.numIters) >> 2],
        stopReason: m.HEAP32[(summary + L.summary.stopReason) >> 2],
        trajectory,
      };
    } finally {
      [x, params, prob, options, summary].forEach((p) => m._free(p));
      m.removeFunction(residualsFn);
      m.removeFunction(gradientFn);
      m.removeFunction(stepFn);
    }
  }
}

/** @param {Function} createTinyopt factory exported by the generated tinyopt.mjs */
export async function createOptimizer(createTinyopt) {
  const module = await createTinyopt();
  return new TinyoptWasmOptimizer(module);
}
