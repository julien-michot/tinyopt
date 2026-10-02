from pathlib import Path

from setuptools import setup


(Path(__file__).resolve().parent / "build" / "pip").mkdir(parents=True, exist_ok=True)
setup()