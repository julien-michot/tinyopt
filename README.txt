
Tinyopt is a minimalist, header-only c++ software component with a modwern interface.

It is designed for the efficient resolution of optimization challenges. Specifically, it targets small-scale, dense non-linear least squares problems, which are prevalent in various scientific and engineering applications.

At its core, Tinyopt is a collection of iterative solvers including Levenberg-Marquardt algorithm, a well-established iterative technique, to navigate the complex landscape of non-linear optimization.

Furthermore, to facilitate the computation of derivatives, a crucial aspect of optimization, `tinyopt` seamlessly integrates the automatic differentiation capabilities.
This integration empowers users to effortlessly compute accurate gradients, thereby streamlining the optimization process and enhancing the overall precision of the solutions obtained.

Here is an example on how to use Tinyopt:

  double x = 1;
  Optimize(x,  [](const auto &x) {return x * x - 2.0;}); // 'x' is √2 afer that, amazing.

Tinyopt is open-source, licensed under the permissive Apache 2.0 License.