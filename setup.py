from setuptools import setup, Extension
import warnings
import sys
import numpy as np

extensions = []
inc_dirs = [np.get_include()]

try:
    import pybind11
except ImportError:
    pybind11 = None
else:
    inc_dirs.append(pybind11.get_include())

# C++ extensions (compiled when pybind11 is available)
thread_flags = [] if sys.platform == "win32" else ["-pthread"]
cpp_exts = [
    Extension(
        "mastermlx.accel._distance_cpp",
        ["mastermlx/accel/_distance_cpp.cpp"],
        include_dirs=inc_dirs,
        extra_compile_args=["-O3", *thread_flags],
        extra_link_args=thread_flags,
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
        extra_compile_args=["-O3", *thread_flags],
        extra_link_args=thread_flags,
        language="c++",
    ),
    Extension(
        "mastermlx.optimize._quad_cpp",
        ["mastermlx/optimize/_quad_cpp.cpp"],
        include_dirs=inc_dirs,
        extra_compile_args=["-O3"],
        language="c++",
    ),
    Extension(
        "mastermlx.graphs._grid_cpp",
        ["mastermlx/graphs/_grid_cpp.cpp"],
        include_dirs=inc_dirs,
        extra_compile_args=["-O3"],
        language="c++",
    ),
    Extension(
        "mastermlx.control._control_cpp",
        ["mastermlx/control/_control_cpp.cpp"],
        include_dirs=inc_dirs,
        extra_compile_args=["-O3", *thread_flags],
        extra_link_args=thread_flags,
        language="c++",
    ),
    Extension(
        "mastermlx.graphs._graph_cpp",
        ["mastermlx/graphs/_graph_cpp.cpp"],
        include_dirs=inc_dirs,
        extra_compile_args=["-O3", *thread_flags],
        extra_link_args=thread_flags,
        language="c++",
    ),
    Extension(
        "mastermlx.ensemble._hist_cpp",
        ["mastermlx/ensemble/_hist_cpp.cpp"],
        include_dirs=inc_dirs,
        extra_compile_args=["-O3"],
        language="c++",
    ),
    Extension(
        "mastermlx.accel._signal_cpp",
        ["mastermlx/accel/_signal_cpp.cpp"],
        include_dirs=inc_dirs,
        extra_compile_args=["-O3", *thread_flags],
        extra_link_args=thread_flags,
        language="c++",
    ),
]
if pybind11 is None:
    warnings.warn(
        "pybind11 is unavailable; C++ extensions will be skipped. "
        "Install pybind11 to enable the C++ backend.",
        RuntimeWarning,
    )
    cpp_exts = []

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
            "mastermlx.accel._conv1d_ops",
            ["mastermlx/accel/_conv1d_ops.pyx"],
            include_dirs=[np.get_include()],
        ),
        Extension(
            "mastermlx.accel._rnn_ops",
            ["mastermlx/accel/_rnn_ops.pyx"],
            include_dirs=[np.get_include()],
        ),
        Extension(
            "mastermlx.accel._signal_ops",
            ["mastermlx/accel/_signal_ops.pyx"],
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
    # Generated C sources can carry NumPy/Cython-specific accessor code from
    # another build environment. Regenerate them against the active headers
    # so source builds remain portable across NumPy and Python versions.
    ], language_level=3, force=True)
    extensions = cpp_exts + cy_exts
except ImportError:
    extensions = cpp_exts

setup(ext_modules=extensions)
