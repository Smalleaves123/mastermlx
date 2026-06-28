from setuptools import setup, Extension
import numpy as np
import pybind11

extensions = []
inc_dirs = [np.get_include(), pybind11.get_include()]

# C++ extensions (always compiled, no Cython needed)
cpp_exts = [
    Extension(
        "mastermlx.accel._distance_cpp",
        ["mastermlx/accel/_distance_cpp.cpp"],
        include_dirs=inc_dirs,
        extra_compile_args=["-O3"],
        language="c++",
    ),
    Extension(
        "mastermlx.accel._kdtree",
        ["mastermlx/accel/_kdtree.cpp"],
        include_dirs=inc_dirs,
        extra_compile_args=["-O3"],
        language="c++",
    ),
    Extension(
        "mastermlx.accel._dtw",
        ["mastermlx/accel/_dtw.cpp"],
        include_dirs=inc_dirs,
        extra_compile_args=["-O3"],
        language="c++",
    ),
]

# Cython extensions (optional)
try:
    from Cython.Build import cythonize
    cy_exts = cythonize([
        Extension(
            "mastermlx.accel._distance_ops",
            ["mastermlx/accel/_distance_ops.pyx"],
            include_dirs=[np.get_include()],
        ),
        Extension(
            "mastermlx.accel._tree_ops",
            ["mastermlx/accel/_tree_ops.pyx"],
            include_dirs=[np.get_include()],
        ),
    ], language_level=3)
    extensions = cpp_exts + cy_exts
except ImportError:
    extensions = cpp_exts

setup(ext_modules=extensions)
