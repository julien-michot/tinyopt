# Copyright 2026 Julien Michot.
# SPDX-License-Identifier: Apache-2.0
"""Installs Tinyopt into a fresh virtualenv and checks that the *installed* package works."""

import subprocess
import sys
import tempfile
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SMOKE = """
import numpy as np, tinyopt
from pathlib import Path
assert "site-packages" in tinyopt.__file__, tinyopt.__file__
assert list(Path(tinyopt.__file__).parent.glob("*tinyopt_c*")), "bundled C library missing"
r = tinyopt.optimize(lambda x: x * x - 2.0, 1.0, log_enabled=False)
assert abs(r.x - 2 ** 0.5) < 1e-6 and r.success, r
t = np.array([1.0, 2.0, 3.0])
for fixed in (True, False):
    r = tinyopt.optimize(lambda x: (x - t, np.eye(3)), np.zeros(3), fixed=fixed, log_enabled=False)
    assert np.allclose(r.x, t, atol=1e-6), r
print("installed tinyopt", tinyopt.__version__, "OK")
"""


def run(cmd, **kw):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.check_call(cmd, **kw)


def main():
    with tempfile.TemporaryDirectory(prefix="tinyopt-pip-") as tmp:
        env_dir = Path(tmp) / "venv"
        # System site-packages provide numpy/pytest so the test needs no network beyond build deps.
        venv.EnvBuilder(system_site_packages=True, with_pip=True, clear=True).create(env_dir)
        py = env_dir / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        run([py, "-m", "pip", "install", "--no-deps", str(ROOT)])
        run([py, "-c", SMOKE], cwd=tmp)  # cwd outside the repo: never import the source tree
        run([py, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(ROOT / "tests" / "python")],
            cwd=tmp)


if __name__ == "__main__":
    main()
