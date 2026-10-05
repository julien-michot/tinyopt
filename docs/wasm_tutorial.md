# Tinyopt WebAssembly tutorial

This guide shows how to compile the C API to WebAssembly and drive it from JavaScript without any
C++ glue code. The example in [examples/wasm](../examples/wasm/README.md) writes the C structs into
module memory and passes JavaScript callbacks into the optimizer.

## 1. Build the wasm module

Use the project environment that includes Emscripten:

```sh
pixi run build-wasm
```

This creates the generated module in `build-wasm/` as `tinyopt.mjs` and `tinyopt.wasm`. The build is
configured for the fixed-size 2D double-precision C API, which keeps the WASM bundle small and fast.

## 2. Use the small JS wrapper

```js
import { TinyoptWasmOptimizer } from './optimizer.js';

const { default: createTinyopt } = await import('./build-wasm/tinyopt.mjs');
const optimizer = new TinyoptWasmOptimizer(await createTinyopt());

const result = optimizer.minimize(problem, [0.0, 0.0], {
  solverType: 0, // Levenberg-Marquardt
  maxIters: 100,
});
```

The wrapper keeps the underlying C ABI intact while exposing plain JavaScript fields such as
`solverType`, `maxIters`, and `logEnabled` instead of raw memory offsets. It is the recommended way
for app code to call the optimizer.

## 3. Define a problem in JavaScript

A 2D residual problem is described by two callbacks:

- `residuals(x, y, r)` fills the residual vector
- `jacobian(x, y, J)` fills the row-major Jacobian matrix, with layout `[dr0/dx, dr0/dy, dr1/dx, dr1/dy]`

```js
const problem = {
  numResiduals: 2,
  residuals: (x, y, r) => {
    r[0] = x + 2 * y - 7;
    r[1] = 2 * x + y - 5;
  },
  jacobian: (x, y, J) => {
    J[0] = 1; J[1] = 2;
    J[2] = 2; J[3] = 1;
  },
};
```

The bundled demo keeps these in [examples/wasm/problems.js](../examples/wasm/problems.js).

## 4. Inspect the result

The wrapper returns a plain JS object with the useful summary data:

```js
const { ok, cost, iters, stopReason, trajectory } = result;
```

`trajectory` contains the points that were accepted by the optimizer, so the interactive example can
plot the path on the cost surface without reading raw memory values.

## 5. Raw ABI (advanced)

If you want to interact with the underlying Emscripten C API directly, the wrapper is still built on
that exact pattern. The generated module exposes the raw C symbols and the usual memory helpers, so
you can write the structs manually:

```js
const x = m._malloc(16);
const params = m._malloc(12);
const problemPtr = m._malloc(16);
const options = m._malloc(1024);
const summary = m._malloc(32);

m.HEAPF64[x >> 3] = 0.0;
m.HEAPF64[(x >> 3) + 1] = 0.0;

m.HEAP32[(params + 0) >> 2] = x;
m.HEAP32[(params + 4) >> 2] = 2;
m.HEAP32[(params + 8) >> 2] = 0;

m.HEAP32[(problemPtr + 0) >> 2] = 1; // type = TINYOPT_EVAL_RESIDUALS
m.HEAP32[(problemPtr + 4) >> 2] = residualsFn;
m.HEAP32[(problemPtr + 8) >> 2] = 2; // num residuals
```

This is the same pattern used by the example code and is useful when you need the exact ABI or are
building a custom binding layer.

### Raw callback wiring

The callback signature is defined by the C API. For a residual-based solver, `tinyopt_problem_t.fn`
points to a function that receives the current point and writes the residuals and optional Jacobian:

```js
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
```

You can also pass a `step_callback` to record every accepted or rejected step:

```js
const stepFn = m.addFunction((dxPtr, dims, isRollback) => {
  // update a JavaScript trajectory here
  return 0;
}, 'iiiii');
```

```js
m._tinyopt_options_default(options);
m.HEAP32[(options + 0) >> 2] = 0; // solver type: LM
m.HEAPU16[(options + 52) >> 1] = 100; // max iterations
m.HEAP32[(options + 104) >> 2] = stepFn;

const status = m._tinyopt_optimize(params, problemPtr, options, summary);
```

The helper `createOptimizer()` in [examples/wasm/optimizer.js](../examples/wasm/optimizer.js) wraps
this pattern and returns the optimization trajectory for plotting and inspection.

## Tips

- Begin with the simple problems in [examples/wasm/problems.js](../examples/wasm/problems.js);
  they are useful sanity checks before trying Rosenbrock or Himmelblau.
- Keep callbacks allocation-free and return quickly: the optimizer calls them repeatedly during the
  nonlinear solve.
- If the callback writes a Jacobian, use the same row-major layout the C API expects.

For the full interactive demo, see [examples/wasm](../examples/wasm/README.md) and run
`pixi run wasm-example`.
