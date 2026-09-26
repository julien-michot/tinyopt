#!/usr/bin/env bun
/**
 * Basic tests for tinyopt JavaScript/WASM bindings
 * Run with: bun test test_basic.js
 */

import { describe, test, expect } from "bun:test";
import { resolve } from "path";

// Import the WASM module
// The path should point to the built tinyopt.js in the build directory
const createTinyoptModule = await import("../../build/tinyopt.js").then(m => m.default);
const tinyopt = await createTinyoptModule();

describe("tinyopt NLLS simple", () => {
  test("should optimize simple residual function", () => {
    // residuals: res = x - [5, 3]
    function resFn(x) {
      return [x[0] - 5.0, x[1] - 3.0];
    }

    // Exercise nested Options.log flag
    const opts = new tinyopt.Options();
    opts.log.print_x = true;

    const x0 = [3.0, 2.0];
    const result = tinyopt.optimize(x0, resFn, null, opts);

    expect(result.output.stop_reason).toBeGreaterThanOrEqual(0);

    // Final x should be closer to target than initial
    const xf = result.x;
    expect(Math.abs(xf[0] - 5.0)).toBeLessThanOrEqual(Math.abs(x0[0] - 5.0));
    expect(Math.abs(xf[1] - 3.0)).toBeLessThanOrEqual(Math.abs(x0[1] - 3.0));

    // Should converge reasonably close to the target
    expect(Math.abs(xf[0] - 5.0)).toBeLessThan(1e-5);
    expect(Math.abs(xf[1] - 3.0)).toBeLessThan(1e-5);
  });
});

describe("tinyopt NLLS with manual gradient", () => {
  test("should optimize with user-provided gradient", () => {
    function resFn(x, grad) {
      const res = [x[0] - 5.0, x[1] - 3.0];
      const cost = res[0] * res[0] + res[1] * res[1]; // squared residuals

      if (grad !== null) {
        // Gradient of squared residuals
        grad[0] = 2.0 * res[0];
        grad[1] = 2.0 * res[1];
      }

      return cost;
    }

    const x0 = [3.0, 2.0];

    // Verify gradient is correct
    const gradOk = tinyopt.check_gradient(x0, resFn, null);
    expect(gradOk).toBe(true);

    // Exercise nested Options.log flag
    const opts = new tinyopt.Options();
    opts.solver_type = tinyopt.Solver.GradientDescent;
    opts.gd.lr = 0.1;
    opts.log.print_x = true;

    const result = tinyopt.optimize(x0, resFn, null, opts);

    expect(result.output.stop_reason).toBeGreaterThanOrEqual(0);

    // Final x should be closer to target than initial
    const xf = result.x;
    expect(Math.abs(xf[0] - 5.0)).toBeLessThanOrEqual(Math.abs(x0[0] - 5.0));
    expect(Math.abs(xf[1] - 3.0)).toBeLessThanOrEqual(Math.abs(x0[1] - 3.0));
  });
});

describe("tinyopt optimization convergence", () => {
  test("should report correct number of iterations", () => {
    function resFn(x) {
      return [x[0] - 1.0, x[1] - 2.0];
    }

    const opts = new tinyopt.Options();
    opts.max_iters = 10;

    const x0 = [0.0, 0.0];
    const result = tinyopt.optimize(x0, resFn, null, opts);

    expect(result.output.num_iters).toBeGreaterThan(0);
    expect(result.output.num_iters).toBeLessThanOrEqual(opts.max_iters);
    // Check that we have residuals count
    expect(result.output.num_residuals).toBeGreaterThanOrEqual(0);
  });
});

describe("tinyopt with custom plus function", () => {
  test("should use custom parameter update", () => {
    function resFn(x) {
      return [x[0] - 5.0, x[1] - 3.0];
    }

    // Custom plus function that just does standard addition
    function customPlus(x, delta) {
      return [x[0] + delta[0], x[1] + delta[1]];
    }

    const opts = new tinyopt.Options();
    opts.max_iters = 50;

    const x0 = [3.0, 2.0];
    const result = tinyopt.optimize(x0, resFn, customPlus, opts);

    expect(result.output.stop_reason).toBeGreaterThanOrEqual(0);

    const xf = result.x;
    expect(Math.abs(xf[0] - 5.0)).toBeLessThan(1e-5);
    expect(Math.abs(xf[1] - 3.0)).toBeLessThan(1e-5);
  });
});
