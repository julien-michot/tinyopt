# Variadic Optimize / operator() contract notes

This note records the behavior that must remain true for all overloads of `Optimize` and `Optimizer_::operator()` when the user passes multiple parameters by reference.

## Required contract

- The arguments are input/output parameters.
- They are passed by non-const reference, and the optimizer updates them in place.
- The callback receives a flattened view of the current parameter state while the optimization loop evaluates candidates.
- After the optimizer terminates, the original user objects must be restored to their final optimized values for the caller to read back.
- The call must be compatible with both scalar objective functions and residual-based NLLS objectives.
- The key regression test is not “must converge” for every simple convex problem; the real contract is “the passed variables are updated and the objective decreases”, because the solver may stop with a success/failure state without a strict `Converged()` flag.

## Why this matters

The callback layer is not a pure wrapper around the user function; it is the handoff between:

1. the flat optimization vector used internally by the solver, and
2. the user-facing parameter objects that live outside the solver.

If the callback restores the wrong value type or restores the wrong live object too early, the optimizer can appear to converge to a wrong local point or can oscillate between states without real progress.

## Failure pattern seen while debugging

The multi-parameter refactor broke in a subtle way:

- the flat vector used by the solver was correct,
- the user callback saw a flattened state, but
- the restoration step could lose the Jet-backed values required by autodiff,
- or it could restore into the wrong local tuple shape after the solver had already moved to a candidate step.

That created false “success” outcomes and a bad local minimum in the `Optimize(x, y, func)` path, especially with automatic differentiation.

## Safe implementation rule

When writing or modifying the variadic overloads:

- build the local evaluation tuple from the current user arguments using the correct cast for the flat value type,
- restore the flat candidate into that local tuple before calling the user callback,
- then call the user function with the local tuple,
- after the loop finishes, restore the final flat state back into the original parameter references.

It is okay to keep the underlying solver flat, but the callback boundary must preserve the original parameter semantics exactly.

## Important gotcha

The optimizer updates the passed variables in place. A test should therefore validate the final state of the arguments, not just whether the callback returned a value or a success flag.

The variadic entry points are not pass-by-value wrappers; they are optimization entry points with live, mutated parameters.

In particular, `Optimize(x, y, func)` and `optimizer(x, y, func)` are in/out APIs. They may return `Succeeded() == true` while `Converged() == false`, and the caller must still read the mutated `x` and `y` values back from the variables themselves.

## Recommendation

Keep the test coverage for variadic overloads localized and explicit. The dedicated regression should check:

- convergence/success result,
- final value of each parameter after optimization,
- both `Optimize(x, y, func)` and `optimizer(x, y, func)` forms,
- and the return of the original variables to their final optimized coordinates.
