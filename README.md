![Tinyopt build linux](https://github.com/julien-michot/tinyopt/actions/workflows/github-actions-build.yml/badge.svg)

# tinyopt
`tinyopt` is a lightweight, header only optimization library.

It can be used to solve small, dense non-linear least squares problems.
It implements a Levenberg-Marquardt optimizer as well as automatic differentiation using Ceres'Jet struct.

# Installation

```shell
git clone https://github.com/julien-michot/tinyopt
cd tinyopt && mkdir build && cd build
cmake ..
make -j && sudo make install
```

Files will be copied to `/usr/include`.

# Usage

Here is how to use `tinyopt` to find the square root of 2.

## Simple API
`Optimize` performs automatic differentiation so you just have to specify the residual(s).

```cpp

  // Define the residuals / loss function, here ε² = ||x*x - 2||²
  auto loss = [](const auto &x) {
    return x * x - 2.0;
  };

  // Define 'x', the parameter to optimize, initialized to '1'
  double x = 1;
  // Optimize!
  const auto &out = Optimize(x, loss);
  // 'x' is now std::sqrt(2.0)
```

## Advanced API
When working with more residuals, `tinyopt` gives some more control,
you can directly accumulate the residuals and jacobians.

```cpp

  // Define 'x', the parameter to optimize, initialized to '1'
  Eigen::Vector<double, 1> x(1);

  // Define the residuals / loss function, here ε² = ||x*x - 2||²
  auto loss = [](const auto &x, auto &JtJ, auto &Jt_res) {
    float res = x * x - 2; // since we want x to be sqrt(2), x*x should be 2
    float J   = 2 * x; // residual's jacobian/derivative w.r.t x
    // Manually update JtJ and Jt*err
    JtJ(0, 0) = J * J;
    Jt_res(0) = J * res; // gradient
    // Return both the squared error and the number of residuals (here, we have only one)
    return std::make_pair(res*res, 1);
  };

  // Setup optimizer options (optional)
  Options options;
  // Optimize!
  Optimize(x, loss, options);
  // 'x' is now std::sqrt(2.0)
```

# Testing

Running the sqrt2 test should give you the following log:

```shell
tinyopt# ./tests/tinyopt_test_sqrt2
✅ #0: X:1.49995 |δX|:5.00e-01 λ:1.00e-04 ⎡σ⎤:0.5000 ε²:1.00000 n:1 dε²:-3.403e+38 ∇ε²:0.000e+00
✅ #1: X:1.41667 |δX|:8.33e-02 λ:3.33e-05 ⎡σ⎤:0.3333 ε²:0.06243 n:1 dε²:-9.376e-01 ∇ε²:0.000e+00
✅ #2: X:1.41422 |δX|:2.45e-03 λ:1.11e-05 ⎡σ⎤:0.3529 ε²:0.00005 n:1 dε²:-6.238e-02 ∇ε²:0.000e+00
✅ #3: X:1.41421 |δX|:2.15e-06 λ:3.70e-06 ⎡σ⎤:0.3536 ε²:0.00000 n:1 dε²:-4.823e-05 ∇ε²:0.000e+00
✅ #4: X:1.41421 |δX|:9.48e-12 λ:1.23e-06 ⎡σ⎤:0.3536 ε²:0.00000 n:1 dε²:-3.702e-11 ∇ε²:0.000e+00
✅ #5: X:1.41421 |δX|:1.57e-16 λ:4.12e-07 ⎡σ⎤:0.3536 ε²:0.00000 n:1 dε²:-7.191e-22 ∇ε²:0.000e+00
❌ #6: X:1.41421 |δX|:1.57e-16 λ:1.37e-07 ε²:0.00000 n:1 dε²:0.000e+00 ∇ε²:0.000e+00
❌ #7: X:1.41421 |δX|:1.57e-16 λ:2.74e-07 ε²:0.00000 n:1 dε²:0.000e+00 ∇ε²:0.000e+00
❌ #8: X:1.41421 |δX|:0.00e+00 λ:5.49e-07 ε²:0.00000 n:1 dε²:0.000e+00 ∇ε²:0.000e+00
===============================================================================
All tests passed (2 assertions in 1 test case)
```


# TODO

Here is what is coming up:

- [x] Support optimizing a single floating point `double x`
- [x] Add auto grad using Ceres's Jet + add simpler API
- [ ] Add examples
- [ ] Add benchmarks
- [ ] Add other methods (e.g. GN, GradDesc)
- [ ] Add more tests (inf, nan, etc.)
- [ ] Add custom parameter struct with manifold example
- [ ] Fix doc

# Citation

If you want, you can cite this work with:

```bibtex
@misc{michot2025,
    author = {Julien Michot},
    title = {tinyopt: A tiny optimization library},
    howpublished = "\url{https://github.com/julien-michot/tinyopt}",
    year = {2025}
}
```

# Contributing
Feel free to contribute to the project, otherwise, have fun using `tinyopt`!