from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup

__version__ = "0.0.0"

ext_modules = [
    Pybind11Extension(
        "pygs",
        ["main.cc"],
        cxx_std=17,
    ),
]

setup(
    name="pygs",
    version=__version__,
    author="Jaesung Park",
    author_email="jaesung.cs@gmail.com",
    ext_modules=ext_modules,
    zip_safe=False,
    python_requires=">=3.10",
)
