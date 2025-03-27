![Tinyopt Builds](https://github.com/julien-michot/tinyopt/actions/workflows/build.yml/badge.svg)

# tinyopt

The `tinyopt` library is a minimalist, header-only c++ software component designed for the efficient resolution of optimization challenges. Specifically, it targets small-scale, dense non-linear least squares problems, which are prevalent in various scientific and engineering applications.

At its core, tinyopt leverages the robust Levenberg-Marquardt algorithm, a well-established iterative technique, to navigate the complex landscape of non-linear optimization. This algorithm, renowned for its ability to strike a balance between the steepest descent and Gauss-Newton methods, ensures reliable convergence even in the presence of challenging problem characteristics.

Furthermore, to facilitate the computation of derivatives, a crucial aspect of optimization, `tinyopt` seamlessly integrates the automatic differentiation capabilities provided by [Ceres'Jet](https://github.com/ceres-solver/ceres-solver). This integration empowers users to effortlessly compute accurate gradients and Hessians, thereby streamlining the optimization process and enhancing the overall precision of the solutions obtained.

# Installation

```shell
git clone https://github.com/julien-michot/tinyopt
cd tinyopt && mkdir build && cd build
cmake ..
make -j && sudo make install
```

Files will be copied to `/usr/include`.

# Usage

Here are a few ways to call `tinyopt`.

## Simple API

`tinyopt` is developer friendly, just call `Optimize` and give it something to optimize, say `x` and something to minimize.

`Optimize` performs automatic differentiation so you just have to specify the residual(s),
no jacodians/derivatives to calculate because you know the pain, right? No pain, vanished, thank you Julien.

### What's the square root of 2?
Beause using `std::sqrt` is over hyped, let's try to recover it using `tinyopt`, here is how to do:

```cpp
// Define 'x', the parameter to optimize, initialized to '1' (yeah, who doesn't like 1?)
double x = 1;
Optimize(x,  [](const auto &x) {return x * x - 2.0;}); // Let's minimize ε = x*x - 2
// 'x' is now std::sqrt(2.0), amazing.
```
That's it. Is it too verbose? Well remove the comments then. Come on, it's just two lines, I can't do better.

### Fitting a circle to a set of points
In this use case, you're given `n` 2D points your job is to fit a circle to them.
Today is your lucky day, `tinyopt` is here to help! Let's see how.

```cpp
Mat2Xf obs(2, n); // fill the observations (n 2D points)

// loss is the sum of || ||p - center||² - radius² ||
auto loss = [&obs]<typename T>(const Eigen::Vector<T, 3> &x) {
  const auto &center = x.template head<2>(); // the first two elements are the cicle position
  const auto radius2 = x.z() * x.z(); // the last one is its radius, taking the square to avoid a sqrt later on
  // Here we compute the squared distances of each point to the center
  auto residuals = (obs.cast<T>().colwise() - center).colwise().squaredNorm();
  // Here we compute the difference of of squared distances are the circle's squared radius
  return (residuals.array() - radius2).matrix().transpose().eval();
  // Make sure the returned type is a scalar or Eigen::Vector<T, N> (thus the .eval())
};


// Define 'x', the parameter to optimize, using the following parametrization: x = {center (x, y), radius}
Vec3 x(0, 0, 1);
Optimize(x, loss); // Optimize!
// 'x' should have converged to a reasonable cicle
std::cout << "Residuals: " << loss(x) << "\n"; // Let's print the final residuals
```

Ok so this is quite more verbose than in the first example but I'm trying to help you understand the syntax, it's easy no?

## Supported Types

With `tinyopt`, you can directly optimize several types of parameters `x`, namely

* `float` or `double` scalar types
* `std:array` of scalars
* `std::vector` of a scalars
* `Eigen::Vector` of fixed or dynamic size
* `Eigen::Matrix` of fixed or dynamic size
* Your custorm class or struct, see below

`tinyopt` will detect whether the size is known at compile time and use optimized data structs to make the optimization faster.

Residuals of the following types can also be returned

* `float` or `double` (or the typename `T`, typically a `Jet<S,N>`)
* `Eigen::Vector` or `Eigen::Matrix`

## Advanced API

### Direct Accumulation, a faster - but manual - way.

When working with more whan one residuals, `tinyopt` allows you to avoid storing a full vector of residuals.
You can directly accumulate the residuals and jacobians this way:

```cpp

// Define 'x', the parameter to optimize, initialized to '1'
double x = 1;

// Define the residuals / loss function, here ε² = ||x*x - 2||²
auto loss = [](const auto &x, auto &JtJ, auto &Jt_res) {
  float res = x * x - 2; // since we want x to be sqrt(2), x*x should be 2
  float J   = 2 * x; // residual's jacobian/derivative w.r.t x
  // Manually update JtJ and Jt*err
  JtJ(0, 0) = J * J;   // normal matrix
  Jt_res(0) = J * res; // gradient (half of it actually)
  // Return both the squared error and the number of residuals (here, we have only one)
  return std::make_pair(res*res, 1);
};

// Setup optimizer options (optional)
Options options;
// Optimize!
const auto &out = Optimize(x, loss, options);
// 'x' is now std::sqrt(2.0), you can check the convergence with out.Converged()
```

`JtJ` and `Jt_res` are the only things you need to update for LM to solve the normal equations and optimize `x`.
It looks a bit rural I know but we can't all live in a city with sleek buidlings,
sometimes it's good to go back to your (square) roots, so take your boots and start coding.

### User defined parameters

Let's say you have a fancy class/struct, say a `Rectangle` and you want to optimize its dimensions.
For instance,

```cpp
template <typename T> // Template is only needed if you need automatic differentiation
struct Rectangle {
  T area() const { return (p2 - p1).norm(); }
  T width() const { return p2.x() - p1.x(); }
  T height() const { return p2.y() - p1.y(); }
  Eigen::Vector<T, 2> center() const { return T(0.5) * (p1 + p2); }
  Eigen::Vector<T, 2> p1, p2; // top left and bottom right positions
};
```

Now if you wan to simply call `Optimize(rectangle, loss)` on your rectangle struct, you need to add
a trait specialization of `params_trait` for you object type:

```cpp

namespace tinyopt::traits { // must be defined in tinyopt::traits

template <typename T>
struct params_trait<Rectangle<T>> {
  using Scalar = T;              // The scalar type
  static constexpr int Dims = 4; // Compile-time parameters dimensions (use Eigen::Dynamic if unknown)
  // Execution-time parameters dimensions [OPTIONAL, if Dims is known)
  static constexpr int dims(const Rectangle<T> &) { return Dims; }

  // Convert a Rectangle to another type 'T2', e.g. T2 = Jet<T> [OPTIONAL, if no Jet]
  // Not needed if you use manual Jacobians instead of automatic differentiation
  template <typename T2> static Rectangle<T2> cast(const Rectangle<T> &rect) {
    return Rectangle<T2>(rect.p1.template cast<T2>(),
                         rect.p2.template cast<T2>());
  }

  // Define how to update the object members (parametrization and manifold)
  static void pluseq(Rectangle<T> &rect,
                     const Eigen::Vector<Scalar, Dims> &delta) {
    // In this case delta is defined as 2 deltas for p1 and p2: [dx1, dy1, dx2, dy2]
    rect.p1 += delta.template head<2>();
    rect.p2 += delta.template tail<2>();
  }
  // Stream operator [OPTIONAL]
  friend std::ostream& operator<<(std::ostream& os, const Rectangle& rect) {
    os << "p1:" << rect.p1.transpose() << ", p2:" << rect.p2.transpose();
    return os;
  }
};

} // namespace tinyopt::traits
```
Now you can start optimizing your custom object, e.g.

```cpp
// Let's say I want the rectangle area to be 10*20, the width = 2 * height and
// the center at (1, 2).
auto loss = [&]<typename T>(const Rectangle<T> &rect) {
  using std::max;
  Eigen::Vector<T, 4> residuals;
  residuals[0] = rect.area() - 10.0f * 20.0f;
  residuals[1] = 100.0f * (rect.width() / max(rect.height(), T(1e-8f)) -
                            2.0f); // the 1e-8 is to prevent division by 0
  residuals.template tail<2>() = rect.center() - Eigen::Vector<T, 2>(1, 2);
  return residuals;
};

Rectangle rectangle;
Optimize(rectangle, loss);
// That's it, rectangle is now fitted to your loss
```

# Testing

`tinyopt` comes with various tests, at least soon enough. Simply run `make test` to run them all.
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

- [ ] Add basic tests (inf, nan, etc.)
- [ ] Add support for e.g. `std::vector<Vec2f> x;`
- [ ] Add support for `Optimize(poses, points, loss);`?
- [ ] Add examples
- [ ] Add benchmarks
- [ ] Add loss (l1, huber, etc.)
- [ ] Add other methods (e.g. GN, GradDesc)
- [ ] Add documentation

Ah ah, you thought I would use Jira for this list? No way.

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