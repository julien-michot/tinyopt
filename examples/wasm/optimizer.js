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

// Populated from the C API headers when the WASM binding is built.
export const LAYOUT = null;
const OPTION_FIELDS = null;
const OPTION_NAMES = null;

function writeOptions(module, options, values, maxConsecutiveFailures) {
  const writes = new Map();
  for (const [name, value] of Object.entries(values)) {
    const field = OPTION_FIELDS[name];
    if (field === undefined) {
      throw new TypeError(`Unknown or unsupported scalar optimizer option: ${name}`);
    }
    const [offset, heap, shift] = field;
    let normalized = offset === LAYOUT.options.maxConsecutiveFailures
      ? maxConsecutiveFailures : value;
    if (typeof normalized === 'boolean') normalized = Number(normalized);
    if (typeof normalized !== 'number' || !Number.isFinite(normalized)) {
      throw new TypeError(`Option ${name} must be a finite number or boolean`);
    }
    writes.set(offset, [heap, shift, normalized]);
  }
  for (const [offset, [heap, shift, value]] of writes) {
    const address = options + offset;
    module[heap][shift === null ? address : address >> shift] = value;
  }
}

const EVAL_RESIDUALS = 1; // tinyopt_eval_type_t
const EVAL_GRADIENT = 2;
// Dogleg, LM and GN take residuals and a Jacobian; the other solvers only use cost and gradient.
const usesResiduals = (solver) => solver <= 1 || solver === 4;
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
    const next = { ...this.options };
    for (const [name, value] of Object.entries(overrides)) {
      const canonicalName = OPTION_NAMES[name];
      next[canonicalName ?? name] = value;
    }
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
    return this._minimizeRaw(solver, problem, x0, y0, config.maxConsecutiveFailures, config);
  }

  _minimizeRaw(solver, problem, x0, y0, maxConsecutiveFailures = 40, optionValues = {}) {
    const m = this.module;
    const L = LAYOUT;
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
    const covarianceBuffer = alloc(4 * Float64Array.BYTES_PER_ELEMENT);
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
      m.HEAP32[(summary + L.summary.finalCovarianceDims) >> 2] = 0;
      m._tinyopt_options_default(options);
      m.HEAP32[(options + L.options.finalCovarianceBuffer) >> 2] = covarianceBuffer;
      m.HEAPU32[(options + L.options.finalCovarianceCapacity) >> 2] = 4;
      writeOptions(m, options, optionValues, effectiveMaxConsecutiveFailures);
      m.HEAP32[(options + L.options.stepCallback) >> 2] = stepFn;
      m.HEAP32[(options + L.options.stepCallbackData) >> 2] = 0;

      const status = m._tinyopt_optimize(params, prob, options, summary);
      const covarianceDims = m.HEAP32[(summary + L.summary.finalCovarianceDims) >> 2];
      return {
        status,
        ok: status === STATUS_OK || status === STATUS_OPTIMIZATION_FAILED,
        cost: m.HEAPF64[(summary + L.summary.finalCost) >> 3],
        iters: m.HEAP32[(summary + L.summary.numIters) >> 2],
        stopReason: m.HEAP32[(summary + L.summary.stopReason) >> 2],
        covariance: covarianceDims > 0
          ? Array.from(
              m.HEAPF64.subarray(
                covarianceBuffer >> 3,
                (covarianceBuffer >> 3) + covarianceDims * covarianceDims,
              ),
            )
          : null,
        trajectory,
      };
    } finally {
      [x, params, prob, options, summary, covarianceBuffer].forEach((p) => m._free(p));
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
