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
             stepCallback: 104, stepCallbackData: 108, logEnabled: 112 },
  // `size` above is a generous upper bound of sizeof(tinyopt_options_t)
  params: { x: 0, dims: 4, plusEq: 8, size: 12 },
  problem: { type: 0, fn: 4, numResiduals: 8, userData: 12, size: 16 },
  summary: { stopReason: 0, numIters: 4, finalCost: 16, size: 32 },
};
const EVAL_RESIDUALS = 1; // tinyopt_eval_type_t
const EVAL_GRADIENT = 2;
// Dogleg, LM and GN take residuals and a Jacobian; the other solvers only use cost and gradient.
const usesResiduals = (solver) => solver <= 1 || solver === 4;
const STATUS_OK = 0;
const STATUS_OPTIMIZATION_FAILED = 3; // Also reported when max_iters is reached.

/** @param {Function} createTinyopt factory exported by the generated tinyopt.mjs */
export async function createOptimizer(createTinyopt) {
  const m = await createTinyopt();
  const L = LAYOUT;
  const alloc = (size) => m._malloc(size);

  return {
    module: m,

    /** Returns some C defaults, to validate LAYOUT. */
    defaultOptions() {
      const options = alloc(L.options.size);
      m._tinyopt_options_default(options);
      const result = {
        maxIters: m.HEAPU16[(options + L.options.maxIters) >> 1],
        maxConsecutiveFailures: m.HEAPU8[options + L.options.maxConsecutiveFailures],
        logEnabled: m.HEAP32[(options + L.options.logEnabled) >> 2],
      };
      m._free(options);
      return result;
    },

    /**
     * Minimizes 0.5 * sum(r_i^2) of `problem` from (x0, y0). The trajectory is rebuilt from the
     * C `step_callback`, which reports each step added to the parameters (and each roll-back).
     * @returns {{status: number, ok: boolean, cost: number, iters: number, stopReason: number,
     *            trajectory: Array<[number, number]>}}
     */
    minimize(solver, problem, x0, y0, maxIters = 100) {
      const residuals = new Float64Array(problem.numResiduals);
      const jacobian = new Float64Array(problem.numResiduals * 2);
      const trajectory = [[x0, y0]];
      let px = x0;
      let py = y0;

      // int residuals(const double *x, int dims, double *r, double **jacobian, int n, void *)
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
      // int acc_grad(const double *x, int dims, double *cost, double *gradient, void *)
      const gradientFn = m.addFunction((xPtr, dims, costPtr, gradPtr) => {
        const x = m.HEAPF64[xPtr >> 3];
        const y = m.HEAPF64[(xPtr >> 3) + 1];
        problem.residuals(x, y, residuals);
        let c = 0;
        for (let i = 0; i < problem.numResiduals; ++i) c += residuals[i] * residuals[i];
        m.HEAPF64[costPtr >> 3] = 0.5 * c;
        if (gradPtr !== 0) { // J^T r
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
      // int step(const float *dx, int dims, int is_rollback, void *)
      const stepFn = m.addFunction((dxPtr, dims, isRollback) => {
        px += m.HEAPF32[dxPtr >> 2];
        py += m.HEAPF32[(dxPtr >> 2) + 1];
        if (isRollback) trajectory.pop(); // back to the previous point, already recorded
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
        // First-order solvers need several step reductions to get below a stable step size
        m.HEAPU8[options + L.options.maxConsecutiveFailures] = 40;
        m.HEAP32[(options + L.options.logEnabled) >> 2] = 0;
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
    },
  };
}
