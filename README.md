![Tinyopt Builds](https://github.com/julien-michot/tinyopt/actions/workflows/build.yml/badge.svg)

![Tinyopt Optimizer](data/tinyopt.jpeg)

# Tinyopt

Is your optimization's convergence rate rivaling the speed of continental drift?

`Tinyopt`, the **header-only C++ hero**, swoops in to save the day! It's like a tiny,
caffeinated mathematician living in your project, ready to efficiently tackle those small-to-large optimization beasties,
including unconstrained and non-linear least squares puzzles.
Perfect for when your science or engineering project is about to implode from too much math.

Tinyopt provides **high-accuracy** and **computationally efficient** optimization capabilities, supporting both dense and sparse problem structures. It can be used on general computers and embedded systems with limited resources and come with a strict memory allocation mode for the hardest but safest systems!
The library integrates a collection of iterative solvers including Gradient Descent, Gauss-Newton and Levenberg-Marquardt, Conjugate Gradient, (l-)BFGS algorithms.

Furthermore, to facilitate the computation of derivatives, `Tinyopt` seamlessly integrates the **automatic differentiation** capabilities which empowers users to effortlessly compute accurate gradients.

Tinyopt is open-source, licensed under the Apache 2.0 License. 🧾

## Why is Tinyopt so fast?

`Tinyopt` achieves efficiency through its Accumulation function, which offers a unique approach.
Instead of the conventional method of storing extensive lists of residuals and their Jacobians,
it empowers users to directly populate the linear system with gradients and Hessians.
This manual filling significantly curtails overhead and memory usage,
requiring storage only for the more compact gradient (and optionally, the Hessian).

Note: even though Tinyopt supports sparse systems, it is not as fast as it could be to optimize large ones, especially block sparse problems. We're still missing some clever tricks to make the optimization fast. It will come so stay (fine) tuned!


## Table of Contents
[Installation](#installation-)

[Usage](#usage-)

[Benchmarks](#benchmarks-how-fast-is-tinyopt-)

[Roadmap](#roadmap-%EF%B8%8F)

[Get Involved](#get-involved--get-in-touch-)

# Installation 📥

## Development Installation (pixi)

We use [pixi](https://pixi.prefix.dev/latest/) as the primary workflow for this repo. It configures the build, manages the environment, and exposes the project’s validation and packaging tasks without repeated direct `cmake` invocations.

```shell
git clone https://github.com/julien-michot/tinyopt
cd tinyopt

# compile and run the full test suite
pixi run tests
pixi run install-local
```

This copies the headers to `/usr/local/include`.

To depend on Tinyopt from another Pixi project, add it to the project dependencies once the package is available in your configured channel:

```toml
[dependencies]
tinyopt = ">=0.1.0"
```

For configuration flags and their defaults, see [CMake Options](docs/guides/cmake-options.md).
For installation and consumer CMake examples, see [Installation and Usage](docs/guides/installation.md).

## Debian / Ubuntu package install

Simply fetch the latest .deb packages in the releases section, then, a simple

```shell
sudo apt-get install *.deb
```
Alternatively, you can build the package with `pixi run build-pkg`.

For release artifacts, version updates, and tagging, see [Packaging and Releasing](docs/development/packaging-and-releasing.md).

## Pip install

For a project-local installation, you can use pip install:

```shell
python -m pip install git+https://github.com/julien-michot/tinyopt.git
# Or, once published on PyPI
python -m pip install tinyopt
```

This installs the headers as a lightweight Python package so the public headers are available alongside the project metadata for later Python binding work.

# Usage 👨🏻‍💻

Explore practical examples from finance, computer vision, biology, astronomy, physics, robotics, and
signal processing in the [C++ domain examples guide](examples/cpp/README.md). Build them with
`pixi run build-examples`.

## Tinyopt: The Easy Way 😎

`Tinyopt` is inspired by the simple syntax of python so it is very developer friendly*, just call `Optimize` and give it something to optimize, say `x` and something to minimize.

`Optimize` performs automatic differentiation so you just have to specify the residual(s),
no jacodians/derivatives to calculate because you know the pain, right? No pain, vanished, thank you Julien.

\* but not compiler friendly, sorry gcc/clang but you'll have to work double because it's all templated.

### Example: What's the square root of 2? 🤓
Beause using `std::sqrt` is over hyped, let's try to recover it using `Tinyopt`, here is how to do:

```cpp
// Import the optimizer you want, say the default NLLS one
using namespace tinyopt::nlls;
// Define 'x', the parameter to optimize, initialized to '1' (yeah, who doesn't like 1?)
double x = 1;
Optimize(x, [](auto &x) { return x * x - 2.0; }); // Let's minimize ε = x*x - 2
// 'x' is now √2, amazing.
```
That's it. Is it too verbose? Well remove the comments then. Come on, it's just two lines, I can't do better.

Running this will give you x = √2 at the end as well as some nerdy info:

```shell
tinyopt# make run_tinyopt_test_sqrt2
💡 #0: τ:0.00ms x:{1} |δx|:5.00e-01 λ:1.00e-04 ε:1.00e+00 n:1 dε:-3.403e+38 |∇|:4.000e+00
✅ #1: τ:0.06ms x:{1.49995} |δx|:8.33e-02 λ:3.33e-05 ε:2.50e-01 n:1 dε:-7.502e-01 |∇|:5.618e-01
✅ #2: τ:0.07ms x:{1.41667} |δx|:2.45e-03 λ:1.11e-05 ε:6.94e-03 n:1 dε:-2.429e-01 |∇|:3.871e-04
✅ #3: τ:0.08ms x:{1.41422} |δx|:2.11e-06 λ:3.70e-06 ε:5.96e-06 n:1 dε:-6.938e-03 |∇|:2.842e-10
✅ #4: τ:0.08ms x:{1.41421} |δx|:4.21e-08 λ:1.23e-06 ε:1.19e-07 n:1 dε:-5.841e-06 |∇|:1.137e-13
🌞 Reached minimal gradient (success)
```

## Tinyopt Example Project

For a minimal CMake consumer example and a step-by-step introduction to the API, see the
[Tinyopt documentation](https://julien-michot.github.io/tinyopt/).

## API Documentation 📚

The project also provides a C ABI with dynamic and generated fixed-size float/double APIs; see the
[C API guide](docs/api/c.md). The WebAssembly example ships with a JS-friendly wrapper that hides the
raw memory layout behind plain `options` and `minimize(problem, start, options)` calls; see
[WebAssembly tutorial](docs/tutorials/wasm.md) and [examples/wasm/README.md](examples/wasm/README.md)
for the walkthrough and live demo (`pixi run wasm-example`).

Examples per language: [C](examples/c/README.md), [C++](examples/cpp/README.md), and
[WebAssembly/JavaScript](examples/wasm/README.md) (optimizer trajectories on 3D cost surfaces).

Browse the [C++ API documentation](https://julien-michot.github.io/tinyopt/api.html) or the
[full documentation](https://julien-michot.github.io/tinyopt/).

Have a look at our [API doc](https://github.com/julien-michot/tinyopt/blob/main/docs/api/cpp.md) or delve into
the full doc at [ReadTheDocs](https://tinyopt.readthedocs.io/en/latest). For local development
workflows, see [this guide](docs/development/benchmarking-and-profiling.md).

# Roadmap 🗺️

Here is what is coming up. Don't trust too much the versions as I go with the flow.

- [ ] Add block sparse solver
- [ ] Fast sparse optimization of large systems (reduced systems)
- [ ] Add fixed and bounded dimensions/parameters
- [ ] Add C, JS, Python bindings
- [ ] Add more robust norms, solvers (Adam, etc) and backend (cuda)
- [ ] Add Newton-Raphson & 3rd order optimizations (Halley/Householder)
- [ ] Add QP/SQP and root finding solvers
- [ ] Add gradient-less optimizations
- [ ] Add more tests, benchmarks.

Ah ah, you thought I would use Jira for this list? No way.

# Get Involved & Get in Touch! 🤝

## Citation 📑

If you find yourself wanting to give us a scholarly nod, feel free to use this BibTeX snippet:

```bibtex
@misc{michot2025,
    author = {Julien Michot},
    title = {tinyopt: A tiny optimization library},
    howpublished = "\url{https://github.com/julien-michot/tinyopt}",
    year = {2025}
}
```

## Fancy Lending a Hand? (We'd Love That!) 🤩
Feel free to contribute to the project, there's plenty of things to add,
from bindings to various languages to adding more solvers, examples and code optimizations
in order to make `Tinyopt`, truly the fastest optimization library!

Otherwise, have fun using `Tinyopt` ;)

## Got Big Ideas (or Just Want to Chat Business)?

If your business needs a super fast 🔥 Bundle Adjustment (BA) or multi-sensor **[SLAM](https://github.com/eta-vision/eta-slam-public)** or if `Tinyopt` is still taking its sweet time with your application and
you're finding yourself drumming your fingers impatiently, don't despair!

Feel free to give [me](https://github.com/julien-michot) a shout, I can probably help!
