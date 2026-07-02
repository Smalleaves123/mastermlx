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
    Extension(
        "mastermlx.accel._kernels_cpp",
        ["mastermlx/accel/_kernels_cpp.cpp"],
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
        Extension(
            "mastermlx.accel._cnn_ops",
            ["mastermlx/accel/_cnn_ops.pyx"],
            include_dirs=[np.get_include()],
        ),
        Extension(
            "mastermlx.control._control_ops",
            ["mastermlx/control/_control_ops.pyx"],
            include_dirs=[np.get_include()],
        ),
        Extension(
            "mastermlx.robotics._robotics_ops",
            ["mastermlx/robotics/_robotics_ops.pyx"],
            include_dirs=[np.get_include()],
        ),
        Extension(
            "mastermlx.math_tools._time_series_ops",
            ["mastermlx/math_tools/_time_series_ops.pyx"],
            include_dirs=[np.get_include()],
        ),
        Extension(
            "mastermlx.control._lqr_ops",
            ["mastermlx/control/_lqr_ops.pyx"],
            include_dirs=[np.get_include()],
        ),
        Extension(
            "mastermlx.robotics._trajectory_ops",
            ["mastermlx/robotics/_trajectory_ops.pyx"],
            include_dirs=[np.get_include()],
        ),
        Extension(
            "mastermlx.estimation._kalman_ops",
            ["mastermlx/estimation/_kalman_ops.pyx"],
            include_dirs=[np.get_include()],
        ),
        Extension(
            "mastermlx.utils._distance_scalar_ops",
            ["mastermlx/utils/_distance_scalar_ops.pyx"],
            include_dirs=[np.get_include()],
        ),
        Extension(
            "mastermlx.utils._kernel_scalar_ops",
            ["mastermlx/utils/_kernel_scalar_ops.pyx"],
            include_dirs=[np.get_include()],
        ),
        Extension(
            "mastermlx.utils._metrics_ops",
            ["mastermlx/utils/_metrics_ops.pyx"],
            include_dirs=[np.get_include()],
        ),
        Extension(
            "mastermlx.estimation._particle_ops",
            ["mastermlx/estimation/_particle_ops.pyx"],
            include_dirs=[np.get_include()],
        ),
    ], language_level=3)
    extensions = cpp_exts + cy_exts
except ImportError:
    extensions = cpp_exts

setup(ext_modules=extensions)
