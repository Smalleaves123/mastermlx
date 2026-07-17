#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

py::tuple quad_gd(Array H_, Array b_, Array x0_, double lr, int max_iter, double tol) {
    auto Hb = H_.request();
    auto bb = b_.request();
    auto xb = x0_.request();
    if (Hb.ndim != 2 || Hb.shape[0] != Hb.shape[1])
        throw std::invalid_argument("H must be a square matrix");
    if (bb.ndim != 1 || xb.ndim != 1 || bb.shape[0] != xb.shape[0] || Hb.shape[0] != xb.shape[0])
        throw std::invalid_argument("H, b, and x0 have incompatible shapes");
    if (xb.shape[0] < 1)
        throw std::invalid_argument("H, b, and x0 must be non-empty");
    if (lr <= 0.0 || max_iter < 1 || tol < 0.0)
        throw std::invalid_argument("lr must be positive, max_iter at least 1, and tol non-negative");

    const auto n = static_cast<py::ssize_t>(xb.shape[0]);
    const auto* H = static_cast<const double*>(Hb.ptr);
    const auto* b = static_cast<const double*>(bb.ptr);
    const auto* x_init = static_cast<const double*>(xb.ptr);
    for (py::ssize_t i = 0; i < n; ++i) {
        if (!std::isfinite(b[i]) || !std::isfinite(x_init[i]))
            throw std::invalid_argument("b and x0 must contain only finite values");
        for (py::ssize_t j = 0; j < n; ++j) {
            if (!std::isfinite(H[i * n + j]))
                throw std::invalid_argument("H must contain only finite values");
        }
    }
    py::array_t<double> x(n);
    py::array_t<double> history(max_iter + 1);
    auto* xp = static_cast<double*>(x.request().ptr);
    auto* hp = static_cast<double*>(history.request().ptr);
    std::vector<double> gradient(static_cast<std::size_t>(n));

    for (py::ssize_t i = 0; i < n; ++i)
        xp[i] = x_init[i];

    auto value = [&]() {
        double out = 0.0;
        for (py::ssize_t i = 0; i < n; ++i) {
            double row = 0.0;
            for (py::ssize_t j = 0; j < n; ++j)
                row += H[i * n + j] * xp[j];
            out += 0.5 * xp[i] * row + b[i] * xp[i];
        }
        return out;
    };

    hp[0] = value();
    int nit = 0;
    bool success = false;
    {
        py::gil_scoped_release release;
        for (int iter = 1; iter <= max_iter; ++iter) {
            double norm = 0.0;
            for (py::ssize_t i = 0; i < n; ++i) {
                double grad = b[i];
                for (py::ssize_t j = 0; j < n; ++j)
                    grad += H[i * n + j] * xp[j];
                gradient[static_cast<std::size_t>(i)] = grad;
                norm = std::max(norm, std::abs(grad));
            }
            if (norm <= tol) {
                nit = iter - 1;
                success = true;
                break;
            }

            double step = 0.0;
            for (py::ssize_t i = 0; i < n; ++i) {
                const double dx = -lr * gradient[static_cast<std::size_t>(i)];
                xp[i] += dx;
                step = std::max(step, std::abs(dx));
            }
            hp[iter] = value();
            nit = iter;
            if (step <= tol) {
                success = true;
                break;
            }
        }
    }

    return py::make_tuple(x, history, nit, success);
}

PYBIND11_MODULE(_quad_cpp, m) {
    m.doc() = "C++ accelerated quadratic optimization";
    m.def("quad_gd", &quad_gd, py::arg("H"), py::arg("b"), py::arg("x0"),
          py::arg("lr"), py::arg("max_iter"), py::arg("tol"));
}
