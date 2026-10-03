# Tinyopt C++ Examples

Build all examples with `pixi run build-examples`. The standalone executables are
written under `build-examples/examples/cpp/`; run any target there to see its
estimated parameters and solver status.

| Domain | Example | Solver | Problem |
| --- | --- | --- | --- |
| Finance | `finance.cpp` | L-BFGS | Long-only mean-variance portfolio fit |
| Computer vision | `computer_vision.cpp` | Gauss-Newton | Pinhole camera calibration |
| Biology | `biology.cpp` | BFGS | Logistic population-growth fit |
| Astronomy | `astronomy.cpp` | Levenberg-Marquardt | One-dimensional astrometry with parallax |
| Physics | `physics.cpp` | Dogleg | Damped oscillator parameter fit |
| Robotics | `robotics.cpp` | Gradient descent | 2D landmark-based translation estimate |
| Signal processing | `signal_processing.cpp` | Conjugate gradient | Sinusoid amplitude and offset fit |

Each program is standalone and uses only Tinyopt and the C++ standard library.