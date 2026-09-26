# JavaScript/WASM Bindings for tinyopt

This directory contains JavaScript/WebAssembly bindings for tinyopt using Emscripten's Embind.

## Building

To build the JavaScript bindings, you need Emscripten installed. Using pixi:

```bash
# Configure and build with Emscripten
pixi run -e javascript configure-js-bindings
pixi run -e javascript build-js-bindings
```

This will generate:
- `build/tinyopt.js` - JavaScript glue code
- `build/tinyopt.wasm` - WebAssembly binary

## Running Tests

The tests use Bun as the test runner:

```bash
pixi run -e javascript test-js-bindings
```

Or manually:
```bash
cd tests/js
bun install
bun test
```

## Usage

### In Node.js or Bun

```javascript
import createTinyoptModule from './build/tinyopt.js';

const tinyopt = await createTinyoptModule();

// Define a residual function
function residuals(x) {
  return [x[0] - 5.0, x[1] - 3.0];
}

// Set up options
const opts = new tinyopt.Options();
opts.max_iterations = 100;
opts.log.print_x = true;

// Optimize
const x0 = [0.0, 0.0];
const result = tinyopt.optimize(x0, residuals, null, opts);

console.log('Final x:', result.x);
console.log('Stop reason:', result.output.stop_reason);
console.log('Iterations:', result.output.num_iterations);
```

### With Manual Gradient

```javascript
function costWithGradient(x, grad) {
  const res = [x[0] - 5.0, x[1] - 3.0];
  const cost = res[0] * res[0] + res[1] * res[1];

  if (grad !== null) {
    grad[0] = 2.0 * res[0];
    grad[1] = 2.0 * res[1];
  }

  return cost;
}

// Check gradient
const gradOk = tinyopt.check_gradient([0.0, 0.0], costWithGradient, null);
console.log('Gradient check:', gradOk);

// Optimize with gradient descent
const opts = new tinyopt.Options();
opts.solver_type = tinyopt.Solver.GradientDescent;
opts.gd.lr = 0.1;

const result = tinyopt.optimize([0.0, 0.0], costWithGradient, null, opts);
```

### Custom Parameter Update (Manifolds)

```javascript
function residuals(x) {
  return [x[0] - 1.0, x[1] - 2.0];
}

// Custom plus function for parameter updates
function customPlus(x, delta) {
  // Example: constrained update (project to unit circle)
  const x_new = [x[0] + delta[0], x[1] + delta[1]];
  const norm = Math.sqrt(x_new[0]**2 + x_new[1]**2);
  return [x_new[0] / norm, x_new[1] / norm];
}

const result = tinyopt.optimize([0.5, 0.5], residuals, customPlus, new tinyopt.Options());
```

## API Reference

### Functions

- `optimize(x0, residuals, plus, opts)`: Main optimization function
  - `x0`: Array - Initial parameter values
  - `residuals`: Function - Residual function `(x) => [...]` or cost function `(x, grad) => cost`
  - `plus`: Function (optional) - Custom parameter update `(x, delta) => x_new`
  - `opts`: Options - Optimization options
  - Returns: `{x: Array, output: Output}` - Optimized parameters and optimization output

- `check_gradient(x0, cost_fn, plus)`: Verify gradient implementation
  - Returns: boolean - true if gradient is correct

### Classes

#### Options
- `solver_type`: Solver enum (LevenbergMarquardt, GaussNewton, GradientDescent)
- `max_iterations`: number
- `cost_tol`: number
- `params_tol`: number
- `grad_tol`: number
- `log`: OptionsLog
- `gd`: OptionsGD

#### Output
- `stop_reason`: StopReason enum
- `num_iterations`: number
- `num_cost_evaluations`: number
- `num_jacobian_evaluations`: number
- `cost_initial`: number
- `cost_final`: number

### Enums

- `Solver`: { LevenbergMarquardt, GaussNewton, GradientDescent }
- `StopReason`: { kNone, kMaxIterations, kCostTol, kParamsTol, kGradTol }
