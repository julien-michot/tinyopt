// Copyright 2026 Julien Michot.
// SPDX-License-Identifier: Apache-2.0

// Headless check of the WebAssembly build: node examples/wasm/test.mjs build-wasm/wasm-example
import assert from 'node:assert/strict';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

import { PROBLEMS, cost } from './problems.js';
import { SOLVERS, createOptimizer } from './optimizer.js';

const dir = path.resolve(process.argv[2] ?? '.');
const { default: createTinyopt } = await import(pathToFileURL(path.join(dir, 'tinyopt.mjs')));
const optimizer = await createOptimizer(createTinyopt);

// The struct layout of optimizer.js must match the C library.
assert.deepEqual(optimizer.defaultOptions(), { maxIters: 50, maxConsecutiveFailures: 5, logEnabled: 1 });

for (const problem of PROBLEMS) {
  // Analytical Jacobian against central finite differences.
  const [x, y] = [problem.start[0] + 0.3, problem.start[1] - 0.2];
  const J = new Float64Array(problem.numResiduals * 2);
  const rp = new Float64Array(problem.numResiduals);
  const rm = new Float64Array(problem.numResiduals);
  problem.jacobian(x, y, J);
  const h = 1e-6;
  for (let i = 0; i < problem.numResiduals; ++i) {
    problem.residuals(x + h, y, rp); problem.residuals(x - h, y, rm);
    const dx = (rp[i] - rm[i]) / (2 * h);
    problem.residuals(x, y + h, rp); problem.residuals(x, y - h, rm);
    const dy = (rp[i] - rm[i]) / (2 * h);
    assert.ok(Math.abs(J[2 * i] - dx) < 1e-4 * (1 + Math.abs(dx)), `${problem.id} dr/dx`);
    assert.ok(Math.abs(J[2 * i + 1] - dy) < 1e-4 * (1 + Math.abs(dy)), `${problem.id} dr/dy`);
  }

  for (const solver of SOLVERS) {
    const [x0, y0] = problem.start;
    const res = optimizer.minimize(solver.id, problem, x0, y0, 200);
    const label = `${problem.id} / ${solver.name}`;
    assert.ok(res.ok, `${label}: status ${res.status}`);
    assert.deepEqual(res.trajectory[0], [x0, y0], `${label}: trajectory starts at x0`);
    const [xe, ye] = res.trajectory.at(-1);
    // Undamped Gauss-Newton may reject every step, leaving only the start point. First-order
    // solvers may end on an unevaluated last step, so only damped solvers must end lower.
    assert.ok(res.trajectory.flat().every(Number.isFinite), `${label}: non-finite point`);
    if (solver.id === 0 || solver.id === 4)
      assert.ok(cost(problem, xe, ye) <= cost(problem, x0, y0), `${label}: cost increased`);
    if (solver.id !== 1) assert.ok(res.trajectory.length >= 2, `${label}: empty trajectory`);
    console.log(`${label}: cost=${res.cost.toExponential(2)} iters=${res.iters} ` +
                `points=${res.trajectory.length}`);
    // LM must reach a minimum from the default starts (Freudenstein-Roth has a local one).
    if (solver.id === 0 && problem.id !== 'freudenstein-roth') {
      assert.ok(res.cost < 1e-6, `${label}: did not converge (cost ${res.cost})`);
    }
  }
}
console.log('wasm example OK');
