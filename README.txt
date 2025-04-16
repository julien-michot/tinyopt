Tinyopt, a lightweight header-only optimization library.

Tinyopt, the header-only C++ hero, swoops in to save the day! It's like a tiny,
caffeinated mathematician living in your project, ready to efficiently tackle those small-to-large optimization beasties,
including unconstrained and non-linear least squares puzzles.
Perfect for when your science or engineering project is about to implode from too much math.

Tinyopt provides high-accuracy and computationally efficient optimization capabilities,
supporting both dense and sparse problem structures.
The library integrates a collection of iterative solvers including Gradient Descent,
Gauss-Newton and Levenberg-Marquardt algorithms (more are coming).

Furthermore, to facilitate the computation of derivatives, Tinyopt seamlessly integrates the
automatic differentiation capabilities which empowers users to effortlessly compute accurate gradients.

Here is an example on how to use Tinyopt:

  using namespace tinyopt::nlls; // Import the optimizer you want, say the default NLLS one
  double x = 1;
  Optimize(x, [](auto &x) { return x * x - 2.0; }); // 'x' is √2 after that, amazing.

Tinyopt is open-source, licensed under the Apache 2.0 License.