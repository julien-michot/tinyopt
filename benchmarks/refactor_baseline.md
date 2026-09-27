# Optimize refactor baseline timings

This file stores the measured baseline for the refactor guardrails on the Optimize dispatcher and variadic parameter path.
Update it after every commit that changes the public Optimize API or optimizer dispatch behavior.

## Commands and measured timings

- `pixi run tests` -> real 2.24 s
- `pixi run bench` -> real 66.99 s

## Result summary

- Test suite: exit 0, 100% tests passed out of 25.
- Benchmark suite: exit 0, 10/10 passed.

## Representative benchmark means

- Prior 50 [AD]: 9.20585 ms
- Prior 50: 3.30463 ms

## Comparison rule

When a refactor changes `include/tinyopt/optimize.h` or `include/tinyopt/optimizers/optimizer.h`, compare the new values against these numbers before claiming parity.
