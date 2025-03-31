
Tired of wrestling with optimization problems that are just a little too big for a napkin sketch?
Tinyopt, the header-only C++ hero, swoops in to save the day! It's like a tiny,
caffeinated mathematician living in your project, ready to efficiently tackle those small-to-large optimization beasties,
including unconstrained and non-linear least squares puzzles.
Perfect for when your science or engineering project is about to implode from too much math.

Tinyopt supports both dense and sparse systems and contains a collection of iterative solvers including Gradient Descent,
Newton and Levenberg-Marquardt algorithms, to navigate the complex landscape of non-linear optimization.

Furthermore, to facilitate the computation of derivatives, a crucial aspect of optimization,
Tinyopt seamlessly integrates the automatic differentiation capabilities.
This integration empowers users to effortlessly compute accurate gradients,
thereby streamlining the optimization process and enhancing the overall precision of the solutions obtained.

Here is an example on how to use Tinyopt:

  double x = 1;
  Optimize(x,  [](const auto &x) {return x * x - 2.0;}); // 'x' is √2 afer that, amazing.

Tinyopt is open-source, licensed under the permissive Apache 2.0 License.