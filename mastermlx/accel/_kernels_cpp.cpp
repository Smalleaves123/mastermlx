#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace py = pybind11;

using Matrix = py::array_t<double, py::array::c_style | py::array::forcecast>;
using Vector = py::array_t<double, py::array::c_style | py::array::forcecast>;

static py::buffer_info require_matrix(const Matrix& value, const char* name) {
    auto info = value.request();
    if (info.ndim != 2) {
        throw std::invalid_argument(std::string(name) + " must be a 2D array");
    }
    if (info.shape[0] <= 0 || info.shape[1] <= 0) {
        throw std::invalid_argument(std::string(name) + " must be non-empty");
    }
    return info;
}

static void require_same_features(
    const py::buffer_info& Xb, const py::buffer_info& Yb) {
    if (Xb.shape[1] != Yb.shape[1]) {
        throw std::invalid_argument("X and Y must have the same number of features");
    }
}

static py::buffer_info require_vector(
    const Vector& value, const char* name, py::ssize_t expected) {
    auto info = value.request();
    if (info.ndim != 1 || info.shape[0] != expected) {
        throw std::invalid_argument(std::string(name) + " must match the number of samples");
    }
    return info;
}

static void require_finite_gamma(double gamma) {
    if (!std::isfinite(gamma)) {
        throw std::invalid_argument("gamma must be finite");
    }
}

static void require_nonnegative(
    const double* data, py::ssize_t size, const char* name) {
    for (py::ssize_t i = 0; i < size; ++i) {
        if (data[i] < 0.0) {
            throw std::invalid_argument(std::string(name) + " expects non-negative inputs");
        }
    }
}

template <typename Fn>
static void parallel_rows(py::ssize_t rows, py::ssize_t cols, Fn&& fn) {
    if (rows < 4 || cols < 1 || static_cast<long double>(rows) * cols < 1048576.0L) {
        fn(0, rows);
        return;
    }
    const unsigned hardware = std::thread::hardware_concurrency();
    const unsigned available = hardware == 0 ? 1u : hardware;
    const unsigned workers = std::min<unsigned>(
        std::min<unsigned>(available, 4u), static_cast<unsigned>(rows));
    if (workers <= 1) {
        fn(0, rows);
        return;
    }
    const py::ssize_t chunk = (rows + workers - 1) / workers;
    std::vector<std::thread> threads;
    threads.reserve(workers);
    for (unsigned worker = 0; worker < workers; ++worker) {
        const py::ssize_t begin = worker * chunk;
        const py::ssize_t end = std::min(rows, begin + chunk);
        if (begin >= end) {
            continue;
        }
        threads.emplace_back([&fn, begin, end]() { fn(begin, end); });
    }
    for (auto& thread : threads) {
        thread.join();
    }
}

// ---------------------------------------------------------------------------
// RBF (Gaussian) kernel — exp(-gamma * ||x-y||^2).
// ---------------------------------------------------------------------------
py::array_t<double> rbf_kernel(Matrix X_, Matrix Y_, double gamma) {
    require_finite_gamma(gamma);
    const auto Xb = require_matrix(X_, "X");
    const auto Yb = require_matrix(Y_, "Y");
    require_same_features(Xb, Yb);
    const auto* X = static_cast<const double*>(Xb.ptr);
    const auto* Y = static_cast<const double*>(Yb.ptr);
    const py::ssize_t n = Xb.shape[0];
    const py::ssize_t m = Yb.shape[0];
    const py::ssize_t d = Xb.shape[1];

    py::array_t<double> out({n, m});
    auto* result = static_cast<double*>(out.request().ptr);
    std::vector<double> xn2(static_cast<std::size_t>(n));
    std::vector<double> yn2(static_cast<std::size_t>(m));
    {
        py::gil_scoped_release release;
        for (py::ssize_t i = 0; i < n; ++i) {
            const double* xi = X + i * d;
            double norm = 0.0;
            for (py::ssize_t k = 0; k < d; ++k) {
                norm += xi[k] * xi[k];
            }
            xn2[static_cast<std::size_t>(i)] = norm;
        }
        for (py::ssize_t j = 0; j < m; ++j) {
            const double* yj = Y + j * d;
            double norm = 0.0;
            for (py::ssize_t k = 0; k < d; ++k) {
                norm += yj[k] * yj[k];
            }
            yn2[static_cast<std::size_t>(j)] = norm;
        }
        parallel_rows(n, m, [&](py::ssize_t begin, py::ssize_t end) {
            for (py::ssize_t i = begin; i < end; ++i) {
                const double* xi = X + i * d;
                double* row = result + i * m;
                for (py::ssize_t j = 0; j < m; ++j) {
                    const double* yj = Y + j * d;
                    double dot = 0.0;
                    for (py::ssize_t k = 0; k < d; ++k) {
                        dot += xi[k] * yj[k];
                    }
                    const double d2 = xn2[static_cast<std::size_t>(i)]
                        + yn2[static_cast<std::size_t>(j)] - 2.0 * dot;
                    row[j] = std::exp(-gamma * std::max(d2, 0.0));
                }
            }
        });
    }
    return out;
}

// RBF fast path using caller-provided squared norms.
py::array_t<double> rbf_kernel_fast(
    Matrix X_, Matrix Y_, Vector xn2_, Vector yn2_, double gamma) {
    require_finite_gamma(gamma);
    const auto Xb = require_matrix(X_, "X");
    const auto Yb = require_matrix(Y_, "Y");
    require_same_features(Xb, Yb);
    const auto xn2b = require_vector(xn2_, "xn2", Xb.shape[0]);
    const auto yn2b = require_vector(yn2_, "yn2", Yb.shape[0]);
    const auto* X = static_cast<const double*>(Xb.ptr);
    const auto* Y = static_cast<const double*>(Yb.ptr);
    const auto* xn2 = static_cast<const double*>(xn2b.ptr);
    const auto* yn2 = static_cast<const double*>(yn2b.ptr);
    const py::ssize_t n = Xb.shape[0];
    const py::ssize_t m = Yb.shape[0];
    const py::ssize_t d = Xb.shape[1];

    py::array_t<double> out({n, m});
    auto* result = static_cast<double*>(out.request().ptr);
    {
        py::gil_scoped_release release;
        parallel_rows(n, m, [&](py::ssize_t begin, py::ssize_t end) {
            for (py::ssize_t i = begin; i < end; ++i) {
                const double* xi = X + i * d;
                double* row = result + i * m;
                for (py::ssize_t j = 0; j < m; ++j) {
                    const double* yj = Y + j * d;
                    double dot = 0.0;
                    for (py::ssize_t k = 0; k < d; ++k) {
                        dot += xi[k] * yj[k];
                    }
                    const double d2 = xn2[i] + yn2[j] - 2.0 * dot;
                    row[j] = std::exp(-gamma * std::max(d2, 0.0));
                }
            }
        });
    }
    return out;
}

py::array_t<double> laplacian_kernel(Matrix X_, Matrix Y_, double gamma) {
    require_finite_gamma(gamma);
    const auto Xb = require_matrix(X_, "X");
    const auto Yb = require_matrix(Y_, "Y");
    require_same_features(Xb, Yb);
    const auto* X = static_cast<const double*>(Xb.ptr);
    const auto* Y = static_cast<const double*>(Yb.ptr);
    const py::ssize_t n = Xb.shape[0];
    const py::ssize_t m = Yb.shape[0];
    const py::ssize_t d = Xb.shape[1];

    py::array_t<double> out({n, m});
    auto* result = static_cast<double*>(out.request().ptr);
    {
        py::gil_scoped_release release;
        parallel_rows(n, m, [&](py::ssize_t begin, py::ssize_t end) {
            for (py::ssize_t i = begin; i < end; ++i) {
                const double* xi = X + i * d;
                double* row = result + i * m;
                for (py::ssize_t j = 0; j < m; ++j) {
                    const double* yj = Y + j * d;
                    double l1 = 0.0;
                    for (py::ssize_t k = 0; k < d; ++k) {
                        l1 += std::abs(xi[k] - yj[k]);
                    }
                    row[j] = std::exp(-gamma * l1);
                }
            }
        });
    }
    return out;
}

py::array_t<double> chi2_kernel(Matrix X_, Matrix Y_, double gamma) {
    require_finite_gamma(gamma);
    const auto Xb = require_matrix(X_, "X");
    const auto Yb = require_matrix(Y_, "Y");
    require_same_features(Xb, Yb);
    const auto* X = static_cast<const double*>(Xb.ptr);
    const auto* Y = static_cast<const double*>(Yb.ptr);
    const py::ssize_t n = Xb.shape[0];
    const py::ssize_t m = Yb.shape[0];
    const py::ssize_t d = Xb.shape[1];
    constexpr double eps = 1e-12;
    require_nonnegative(X, n * d, "chi2_kernel");
    require_nonnegative(Y, m * d, "chi2_kernel");

    py::array_t<double> out({n, m});
    auto* result = static_cast<double*>(out.request().ptr);
    {
        py::gil_scoped_release release;
        parallel_rows(n, m, [&](py::ssize_t begin, py::ssize_t end) {
            for (py::ssize_t i = begin; i < end; ++i) {
                const double* xi = X + i * d;
                double* row = result + i * m;
                for (py::ssize_t j = 0; j < m; ++j) {
                    const double* yj = Y + j * d;
                    double chi2 = 0.0;
                    for (py::ssize_t k = 0; k < d; ++k) {
                        const double diff = xi[k] - yj[k];
                        const double den = xi[k] + yj[k];
                        chi2 += diff * diff / std::max(den, eps);
                    }
                    row[j] = std::exp(-gamma * 0.5 * chi2);
                }
            }
        });
    }
    return out;
}

py::array_t<double> additive_chi2_kernel(Matrix X_, Matrix Y_) {
    const auto Xb = require_matrix(X_, "X");
    const auto Yb = require_matrix(Y_, "Y");
    require_same_features(Xb, Yb);
    const auto* X = static_cast<const double*>(Xb.ptr);
    const auto* Y = static_cast<const double*>(Yb.ptr);
    const py::ssize_t n = Xb.shape[0];
    const py::ssize_t m = Yb.shape[0];
    const py::ssize_t d = Xb.shape[1];
    constexpr double eps = 1e-12;
    require_nonnegative(X, n * d, "additive_chi2_kernel");
    require_nonnegative(Y, m * d, "additive_chi2_kernel");

    py::array_t<double> out({n, m});
    auto* result = static_cast<double*>(out.request().ptr);
    {
        py::gil_scoped_release release;
        parallel_rows(n, m, [&](py::ssize_t begin, py::ssize_t end) {
            for (py::ssize_t i = begin; i < end; ++i) {
                const double* xi = X + i * d;
                double* row = result + i * m;
                for (py::ssize_t j = 0; j < m; ++j) {
                    const double* yj = Y + j * d;
                    double acc = 0.0;
                    for (py::ssize_t k = 0; k < d; ++k) {
                        const double den = xi[k] + yj[k];
                        acc += 2.0 * xi[k] * yj[k] / std::max(den, eps);
                    }
                    row[j] = acc;
                }
            }
        });
    }
    return out;
}

py::array_t<double> hellinger_kernel(Matrix X_, Matrix Y_) {
    const auto Xb = require_matrix(X_, "X");
    const auto Yb = require_matrix(Y_, "Y");
    require_same_features(Xb, Yb);
    const auto* X = static_cast<const double*>(Xb.ptr);
    const auto* Y = static_cast<const double*>(Yb.ptr);
    const py::ssize_t n = Xb.shape[0];
    const py::ssize_t m = Yb.shape[0];
    const py::ssize_t d = Xb.shape[1];
    require_nonnegative(X, n * d, "hellinger_kernel");
    require_nonnegative(Y, m * d, "hellinger_kernel");

    py::array_t<double> out({n, m});
    auto* result = static_cast<double*>(out.request().ptr);
    {
        py::gil_scoped_release release;
        parallel_rows(n, m, [&](py::ssize_t begin, py::ssize_t end) {
            for (py::ssize_t i = begin; i < end; ++i) {
                const double* xi = X + i * d;
                double* row = result + i * m;
                for (py::ssize_t j = 0; j < m; ++j) {
                    const double* yj = Y + j * d;
                    double acc = 0.0;
                    for (py::ssize_t k = 0; k < d; ++k) {
                        acc += std::sqrt(xi[k]) * std::sqrt(yj[k]);
                    }
                    row[j] = acc;
                }
            }
        });
    }
    return out;
}

PYBIND11_MODULE(_kernels_cpp, m) {
    m.doc() = "C++ accelerated kernel functions";
    m.def("rbf_kernel", &rbf_kernel);
    m.def("rbf_kernel_fast", &rbf_kernel_fast);
    m.def("laplacian_kernel", &laplacian_kernel);
    m.def("chi2_kernel", &chi2_kernel);
    m.def("additive_chi2_kernel", &additive_chi2_kernel);
    m.def("hellinger_kernel", &hellinger_kernel);
}
