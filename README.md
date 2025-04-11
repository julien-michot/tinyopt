![Tinyopt Builds](https://github.com/julien-michot/tinyopt/actions/workflows/build.yml/badge.svg)
![Tinyopt-example Builds](https://github.com/julien-michot/tinyopt-example/actions/workflows/build.yml/badge.svg)

# Tinyopt

Tired of wrestling with optimization problems that are just a little too big for a napkin sketch?
`Tinyopt`, the header-only C++ hero, swoops in to save the day! It's like a tiny,
caffeinated mathematician living in your project, ready to efficiently tackle those small-to-large optimization beasties,
including unconstrained and non-linear least squares puzzles.
Perfect for when your science or engineering project is about to implode from too much math.

Tinyopt supports both dense and sparse systems and contains a collection of iterative solvers including Gradient Descent,
Gauss-Newton and Levenberg-Marquardt algorithms (more are coming).

Furthermore, to facilitate the computation of derivatives, `tinyopt` seamlessly integrates the automatic differentiation capabilities which empowers users to effortlessly compute accurate gradients.


## Table of Contents
[Installation](#installation)

[Usage](#usage)

[Roadmap](#roadmap)

[Contributing](#contributing)

# Installation

```shell
git clone https://github.com/julien-michot/tinyopt
cd tinyopt && mkdir build && cd build
cmake ..
make -j && sudo make install
```

Files will be copied to `/usr/include`.

# Usage

## Tinyopt: The "Just Works" Example (Minimal Edition)

Feeling lost? Fear not! We've crafted a delightful, teeny-tiny CMake project in [tinyopt-example](https://github.com/julien-michot/tinyopt-example) that'll have you parsing options faster than you can say "command-line arguments." It's so simple, even your pet rock could probably figure it out. (Though, we haven't tested that rigorously.)

## Tinyopt: the easy way

`tinyopt` is inspired by the simple syntax of python so it is very developer friendly*, just call `Optimize` and give it something to optimize, say `x` and something to minimize.

`Optimize` performs automatic differentiation so you just have to specify the residual(s),
no jacodians/derivatives to calculate because you know the pain, right? No pain, vanished, thank you Julien.

\* but not compiler friendly, sorry gcc/clang but you'll have to work double because it's all templated.

### What's the square root of 2?
Beause using `std::sqrt` is over hyped, let's try to recover it using `tinyopt`, here is how to do:

```cpp
// Define 'x', the parameter to optimize, initialized to '1' (yeah, who doesn't like 1?)
double x = 1;
Optimize(x,  [](const auto &x) {return x * x - 2.0;}); // Let's minimize ε = x*x - 2
// 'x' is now √2, amazing.
```
That's it. Is it too verbose? Well remove the comments then. Come on, it's just two lines, I can't do better.


### API Documentation

Have a look at our [API doc](https://github.com/julien-michot/tinyopt/blob/main/docs/API.md) or delve into
the full doc at [ReadTheDocs](https://tinyopt.readthedocs.io/en/latest).

## Testing

`tinyopt` comes with various tests, at least soon enough. Simply run `make test` to run them all.
Running the sqrt2 test should give you the following log:

```shell
tinyopt# ./tests/tinyopt_test_sqrt2
✅ #0: x:1.49995 |δx|:5.00e-01 λ:1.00e-04 ⎡σ⎤:0.5000 ε:1.00000 n:1 dε:-3.403e+38 ∇ε:0.000e+00
✅ #1: x:1.41667 |δx|:8.33e-02 λ:3.33e-05 ⎡σ⎤:0.3333 ε:0.06243 n:1 dε:-9.376e-01 ∇ε:0.000e+00
✅ #2: x:1.41422 |δx|:2.45e-03 λ:1.11e-05 ⎡σ⎤:0.3529 ε:0.00005 n:1 dε:-6.238e-02 ∇ε:0.000e+00
✅ #3: x:1.41421 |δx|:2.15e-06 λ:3.70e-06 ⎡σ⎤:0.3536 ε:0.00000 n:1 dε:-4.823e-05 ∇ε:0.000e+00
✅ #4: x:1.41421 |δx|:9.48e-12 λ:1.23e-06 ⎡σ⎤:0.3536 ε:0.00000 n:1 dε:-3.702e-11 ∇ε:0.000e+00
✅ #5: x:1.41421 |δx|:1.57e-16 λ:4.12e-07 ⎡σ⎤:0.3536 ε:0.00000 n:1 dε:-7.191e-22 ∇ε:0.000e+00
❌ #6: x:1.41421 |δx|:1.57e-16 λ:1.37e-07 ε:0.00000 n:1 dε:0.000e+00 ∇ε:0.000e+00
❌ #7: x:1.41421 |δx|:1.57e-16 λ:2.74e-07 ε:0.00000 n:1 dε:0.000e+00 ∇ε:0.000e+00
❌ #8: x:1.41421 |δx|:0.00e+00 λ:5.49e-07 ε:0.00000 n:1 dε:0.000e+00 ∇ε:0.000e+00
☀ Reached minimal gradient (success)
===============================================================================
All tests passed (2 assertions in 1 test case)
```


## Dependencies

We currently only depends on the amazing [Eigen](https://gitlab.com/libeigen/eigen) library, that's it!
Automatic differentiation is done using [Ceres solver](http://ceres-solver.org/)'s Jet but we cloned
and patched it locally so no need to install Ceres.

# Roadmap

Here is what is coming up. Don't trust too much the versions as I go with the flow.

### v1 (stable API + Armadillo)

- [ ] Add l-BFGS for large sparse problems
- [ ] Native support of Armadillo (as alternative to Eigen)
- [ ] Refactor Solvers
- [ ] Update all docs

### v1.x (Bindings)
- [ ] Add C API
- [ ] Add python binding
- [ ] Add Rust binding

### v2 (refactoring, speed-ups & many solvers)
- [ ] Speed-up compilation (e.g. c++20 Concepts)
- [ ] Add various more solvers (CG, Adam, ...) and backend (e.g. Cuda)
- [ ] Speed-up large problems (e.g. AMD)

Ah ah, you thought I would use Jira for this list? No way.

# Get Involved & Get in Touch!

## Citation

If you find yourself wanting to give us a scholarly nod, feel free to use this fancy BibTeX snippet:

```bibtex
@misc{michot2025,
    author = {Julien Michot},
    title = {tinyopt: A tiny optimization library},
    howpublished = "\url{https://github.com/julien-michot/tinyopt}",
    year = {2025}
}
```

## Fancy Lending a Hand? (We'd Love That!)
Feel free to contribute to the project, there's plenty of things to add,
from bindings to various languages to adding more solvers, examples and code optimizations
in order to make `tinyopt`, truely the fastest optimization library!

Otherwise, have fun using `tinyopt` ;)

## Got Big Ideas (or Just Want to Chat Business)?

If `tinyopt` is still taking its sweet time with your application and you're finding yourself drumming your fingers impatiently, don't despair!
Feel free to give me a shout @julien-michot.
I might just have a few more optimization rabbits I can pull out of my hat (or, you know, my code editor).
Let's see if we can inject a little more pep into its step!
