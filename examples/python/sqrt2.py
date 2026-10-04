# Copyright 2026 Julien Michot.
# SPDX-License-Identifier: Apache-2.0
"""What's the square root of 2? Python scalar in, Python float out."""

import tinyopt

res = tinyopt.optimize(lambda x: x * x - 2.0, 1.0)
print(f"sqrt(2) = {res.x:.9f} after {res.iters} iterations ({res.stop_reason.name})")
assert abs(res.x - 2**0.5) < 1e-6
