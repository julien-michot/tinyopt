#include <tinyopt/tinyopt.h>
#include <cmath>

int main() {
  double x = 1.0;
  tinyopt::Optimize(x, [](const auto &xi) { return xi * xi - 2.0; });
  return std::abs(x - std::sqrt(2.0)) < 1e-4 ? 0 : 1;
}
