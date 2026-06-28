from setuptools import setup, Extension
import numpy as np

extensions = []

# C++ extensions (always compiled, no Cython needed)
cpp_exts = [
    Extension(
        "mastermlx.accel._distance_cpp",
        ["mastermlx/accel/_distance.cpp"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-march=native"],
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
