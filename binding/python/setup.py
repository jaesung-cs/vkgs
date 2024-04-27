import os

from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup

__version__ = "0.0.0"

root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

ext_modules = [
    Pybind11Extension(
        "_pygs_cpp",
        ["pygs_cpp/main.cc"],
        include_dirs=[
            os.path.join(root, "include"),
        ],
        library_dirs=[
            os.path.join(root, "build"),
            os.path.join(root, "build", "Release"),
        ],
        libraries=[
            "pygs",
        ],
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
    packages=["pygs"],
)
