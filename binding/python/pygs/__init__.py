
import os
import sys

root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if sys.platform.startswith("win"):
    dll_dir = os.path.abspath(os.path.join(root, "build", "Release"))
    print(f"[pygs] [Windows] adding dll directory: {dll_dir}")
    os.add_dll_directory(dll_dir)

import _pygs_cpp as _C  # noqa


def show():
    _C.show()


def load(ply_filepath):
    ply_filepath = os.path.abspath(ply_filepath)
    print(f"load {ply_filepath}")
    _C.load(ply_filepath)


def close():
    _C.close()


__all__ = ["show", "load", "close"]
