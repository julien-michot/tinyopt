# Optimize refactor plan and implementation notes

This document captures the actual refactor path that proved reliable in this codebase, the exact failure modes we hit during the first attempts, and the staged plan for the next refactor pass.

## 1. Goal

The refactor target is narrow and deliberate:

- clean up the public `Optimize(...)` entry points without changing their user-facing semantics
- extend the overload family to support multiple parameters, while preserving the single-parameter case
- align the optimizer instance call path (`operator()`) with the same contract
- make the shared logic inside `include/tinyopt/optimize.h` and `include/tinyopt/optimizers/optimizer.h` easier to reason about
- keep runtime and compile-time behavior inside the existing guardrails

This is not a solver rewrite. The solver math remains stable; the refactor must be about dispatch, adaptation, and parameter packing.

## 2. Hard constraints from the real codebase

### 2.1 API compatibility

The public API must stay compatible:

- `Optimize(x, func, options)` keeps working exactly as before
- `Optimize(x, y, func, options)` and larger variadic overloads are additive only
- the `operator()(x, func)` family on optimizer objects stays valid
- the overloads must allow both scalar objective functions and residual-based NLLS callables

No silent API break is acceptable.

### 2.2 Parameter semantics

The parameters passed by the user are in/out values. They are mutated by optimizer updates and must be restored to the caller-owned objects after the optimization run.

The correct high-level contract is:

- flatten the user parameters into one internal vector for the solver
- evaluate the objective against a reconstructed tuple of the caller values
- restore the result back into the original parameter objects

This is the core invariant that must never be violated.

### 2.3 Runtime guardrail

The refactor must keep runtime behavior effectively unchanged.

The barrier is simple:

- if the non-Ceres benchmark suite gets materially slower, the refactor stops
- if the public API changes beyond the overload extension, the refactor stops
- if the compile path becomes more fragile without a clear benefit, the refactor stops

Benchmark parity is the gate, not a suggestion.

## 3. What we tried and why it failed

### 3.1 First attempted simplification: centralize the variadic wrapper in the shared helper path

The attempted change tried to normalize the multi-parameter path into a single reusable helper that created:

- a flat vector
- a wrapped callable
- a shared tuple reconstruction step

This was conceptually appealing and looked like the cleanest route, but the implementation failed at the template boundary between:

- the internal flat `Eigen::Vector` used by the solver
- the reconstructed local tuple built from the user parameters
- the restored values that are expected to be available to the objective callback

The failure was not mathematical; it was an overload and reconstruction bug. The wrapper ended up reconstructing the wrong value shape or carrying the wrong typed state across the callback boundary, which by design is the most fragile part of the code.

### 3.2 Why the first attempt broke the code

The actual root cause was that the centralized call wrapper changed the contract in a subtle way:

- the flat vector was still the solver state, but the callback executed against a reconstructed tuple that was not always identically typed to the original data
- the restoration step was too eager or not synchronized with the specific parameter pack layout
- the wrapper function was too generic and lost some of the original parameter identity that the downstream objective depends on

In short, the refactor centralized the logic too early, before the caller-owned parameter semantics had been made explicit. Once that happened, the code compiled in some cases and broke in others depending on mixed scalar/vector/matrix parameter shapes.

### 3.3 What that taught us

The real invariant is not just “flatten and restore” in the abstract. It is:

- flatten exactly the parameter values that the solver is allowed to manipulate
- reconstruct exactly the tuple of caller-owned parameters before objective evaluation
- restore the final candidate back to the original arguments in the same order they were flattened
- keep the public API and the optimizer `operator()` entry points using the same contract

This is the contract we must preserve during any later refactor.

## 4. The right architecture to refactor toward

### 4.1 Keep the flat-vector model, but make it internal-only

The solver should continue to operate on a flat parameter vector. That is a valid implementation detail and should stay that way unless there is strong evidence otherwise.

The important piece is that the solver and the callback boundary remain separate:

- solver state = flat vector
- user callback state = original parameter tuple reconstructed just before evaluation
- external user objects = the actual caller-owned values that are updated in place

This separation is the stable design.

### 4.2 Decide on one central dispatcher policy

The public `Optimize(...)` overloads and the optimizer object `operator()` family should share the same normalization and dispatch policy.

The refactor should aim for a shared internal flow like this:

1. accept a parameter pack
2. flatten the parameter pack to a single vector
3. create a callback wrapper that reconstructs the original parameter layout
4. call the underlying solver with the flat vector and wrapped cost function
5. restore the final values back into the caller-owned objects
6. return the `Output`

That policy should live in a common helper path, but the actual behavior must remain faithful to the original solver contract.

## 5. Refactor plan: Stage A, B, C

### Stage A — freeze the behavior and capture a baseline

This is the guardrail step before any larger reshuffle.

Required work:

- run the existing full test suite
- run the benchmark suite
- record compile-time metrics for the test target
- save the runtime delta against the current baseline

Goal:

- confirm the current implementation is correct
- preserve the evidence needed for future comparison
- keep a clean, trusted “before” state

This is mandatory before any bigger cleanup.

### Stage B — refactor the dispatcher and optimizer call path

This is the first actual refactor stage.

Focus:

- `include/tinyopt/optimize.h`
- `include/tinyopt/optimizers/optimizer.h`
- the multi-parameter overloads in both public and object-level APIs

What to clean up:

- reduce duplicated flatten/restore logic
- standardize the order of parameter restoration and callback reconstruction
- keep the wrapper logic exact to the same parameter pack order used in flattening
- ensure the same logic is used both by `Optimize(...)` and by `Optimizer_::operator()`

Important rule:

- do not rewrite solver math
- do not change public behavior except adding overload support for more parameters
- do not centralize the logic in a more generic form until the contract is proven stable

### Stage C — normalize callable shape and trait dispatch

Only after Stage B is green and benchmark-safe should the traits/callable-shape layer be cleaned up.

This stage should focus on clarifying the shape of the objective, not on changing the solver or the user-facing dispatch.

Target behavior:

- scalar objective functions
- residual-returning NLLS callables
- manual accumulation functions with gradient only
- manual accumulation functions with gradient and Hessian
- dense or sparse output paths when applicable

Recommended internal structure:

- helper predicates based on `std::invoke_result_t`
- a central call-shape classification helper
- a small classification enum or trait switch, not ad hoc branching in multiple places

This is where the code becomes easier to maintain—but not before Stage B is stabilized.

### Stage D — naming and cleanup pass

Only once the behavior is stable and benchmark-safe:

- rename internal helpers to express mathematical meaning clearly
- unify names across `Optimize`, call wrappers, and optimizer internals
- simplify repeated patterns without changing behavior

This stage is optional cleanup; it is not the place to change the algorithm or the API.

## 6. Refactor rules we must keep

1. Keep the single-parameter path working exactly as it does now.
2. Add multi-parameter overloads without replacing the old ones.
3. Keep the optimizer object and the global helper synchronized.
4. The flatten/restore boundary is a hard requirement, not an implementation detail to ignore.
5. Benchmark/runtime parity is a blocker; compile-time reduction is a bonus.
6. Every larger refactor must be validated with tests and benchmarks before moving on.

## 7. What we should do next

The next session should proceed in this order:

1. start from the current green baseline
2. isolate the multi-parameter wrapper contract in one focused regression test
3. refactor the shared entry-point logic in `include/tinyopt/optimize.h`
4. refactor the operator path in `include/tinyopt/optimizers/optimizer.h`
5. only then normalize callable classification in the traits layer
6. re-run tests and compare benchmarks against baseline

This sequence keeps the refactor honest and avoids repeating the mistake of centralizing too much too early.

## 8. Required before/after comparison table

For every refactor, record the numbers in a simple table before and after the patch. Do not claim parity or a speedup without both entries.

| Metric | Before refactor | After refactor | Delta | Status |
| --- | ---: | ---: | ---: | --- |
| Compile time (test target) | 2.24 s real (`pixi run tests`) | value | difference | faster / slower / neutral |
| Benchmark runtime (non-Ceres) | 66.99 s real (`pixi run bench`) | value | difference | faster / slower / neutral |

This is mandatory proof for every refactor. If the before value was not recorded, the refactor is incomplete for runtime proof.

## 9. Final guideline for future work

The main lesson from the failed attempt is simple:

- the implementation is not failing because the solver does not work
- the implementation is failing because the callback boundary between flat internal state and caller-owned parameter objects is too easy to break when the refactor becomes overly generic

The future refactor must be disciplined:

- keep the solver state flat
- keep the callback state reconstructed and typed
- restore before returning
- share the same dispatcher logic between `Optimize` and optimizer `operator()`
- do not let code cleanup outrun behavioral proof

That is the version of the refactor that we should attempt next.
