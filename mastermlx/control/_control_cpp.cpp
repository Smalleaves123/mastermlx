#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <stdexcept>

namespace py = pybind11;
using Matrix = py::array_t<double, py::array::c_style | py::array::forcecast>;

py::array_t<double> linear_rollout(Matrix A_, Matrix B_, Matrix x0_, Matrix U_) {
    const auto A = A_.request();
    const auto B = B_.request();
    const auto x0 = x0_.request();
    const auto U = U_.request();
    if (A.ndim != 2 || A.shape[0] != A.shape[1]) {
        throw std::invalid_argument("A must be a square 2D matrix");
    }
    const auto n = A.shape[0];
    if (B.ndim != 2 || B.shape[0] != n || B.shape[1] < 1) {
        throw std::invalid_argument("B must have shape (A.shape[0], control_dim)");
    }
    const auto m = B.shape[1];
    if (x0.ndim != 1 || x0.shape[0] != n) {
        throw std::invalid_argument("x0 must have shape (A.shape[0],)");
    }
    if (U.ndim != 2 || U.shape[1] != m) {
        throw std::invalid_argument("U must have shape (steps, B.shape[1])");
    }

    const auto* a = static_cast<const double*>(A.ptr);
    const auto* b = static_cast<const double*>(B.ptr);
    const auto* initial = static_cast<const double*>(x0.ptr);
    const auto* controls = static_cast<const double*>(U.ptr);
    const py::ssize_t steps = U.shape[0];
    py::array_t<double> states({steps + 1, n});
    auto* output = static_cast<double*>(states.request().ptr);
    {
        py::gil_scoped_release release;
        for (py::ssize_t i = 0; i < n; ++i) {
            output[i] = initial[i];
        }
        for (py::ssize_t t = 0; t < steps; ++t) {
            const auto* current = output + t * n;
            auto* next = output + (t + 1) * n;
            const auto* control = controls + t * m;
            for (py::ssize_t i = 0; i < n; ++i) {
                double value = 0.0;
                for (py::ssize_t j = 0; j < n; ++j) {
                    value += a[i * n + j] * current[j];
                }
                for (py::ssize_t j = 0; j < m; ++j) {
                    value += b[i * m + j] * control[j];
                }
                next[i] = value;
            }
        }
    }
    return states;
}

PYBIND11_MODULE(_control_cpp, m) {
    m.doc() = "C++ kernels for callback-free control workloads";
    m.def("linear_rollout", &linear_rollout, py::arg("A"), py::arg("B"), py::arg("x0"), py::arg("U"));
}
