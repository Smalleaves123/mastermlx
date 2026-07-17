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
// Pairwise squared Euclidean — O(n*m*d), no intermediate 3D array.
// ---------------------------------------------------------------------------
py::array_t<double> pairwise_squared_euclidean(Matrix X_, Matrix Y_) {
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
                    double acc = 0.0;
                    for (py::ssize_t k = 0; k < d; ++k) {
                        const double diff = xi[k] - yj[k];
                        acc += diff * diff;
                    }
                    row[j] = acc;
                }
            }
        });
    }
    return out;
}

// Direct one-pass implementation avoids the temporary squared-distance array.
py::array_t<double> pairwise_distances(Matrix X_, Matrix Y_) {
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
                    double acc = 0.0;
                    for (py::ssize_t k = 0; k < d; ++k) {
                        const double diff = xi[k] - yj[k];
                        acc += diff * diff;
                    }
                    row[j] = std::sqrt(std::max(acc, 0.0));
                }
            }
        });
    }
    return out;
}

py::array_t<double> pairwise_manhattan(Matrix X_, Matrix Y_) {
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
                    double acc = 0.0;
                    for (py::ssize_t k = 0; k < d; ++k) {
                        acc += std::abs(xi[k] - yj[k]);
                    }
                    row[j] = acc;
                }
            }
        });
    }
    return out;
}

py::array_t<double> pairwise_chebyshev(Matrix X_, Matrix Y_) {
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
                    double best = 0.0;
                    for (py::ssize_t k = 0; k < d; ++k) {
                        best = std::max(best, std::abs(xi[k] - yj[k]));
                    }
                    row[j] = best;
                }
            }
        });
    }
    return out;
}

py::array_t<double> pairwise_minkowski(Matrix X_, Matrix Y_, double p) {
    if (!std::isfinite(p) || p <= 0.0) {
        throw std::invalid_argument("p must be positive and finite");
    }
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
                    double acc = 0.0;
                    for (py::ssize_t k = 0; k < d; ++k) {
                        acc += std::pow(std::abs(xi[k] - yj[k]), p);
                    }
                    row[j] = std::pow(acc, 1.0 / p);
                }
            }
        });
    }
    return out;
}

py::array_t<double> pairwise_canberra(Matrix X_, Matrix Y_) {
    const auto Xb = require_matrix(X_, "X");
    const auto Yb = require_matrix(Y_, "Y");
    require_same_features(Xb, Yb);
    const auto* X = static_cast<const double*>(Xb.ptr);
    const auto* Y = static_cast<const double*>(Yb.ptr);
    const py::ssize_t n = Xb.shape[0];
    const py::ssize_t m = Yb.shape[0];
    const py::ssize_t d = Xb.shape[1];
    constexpr double eps = 1e-12;

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
                        const double num = std::abs(xi[k] - yj[k]);
                        const double den = std::abs(xi[k]) + std::abs(yj[k]);
                        acc += num / std::max(den, eps);
                    }
                    row[j] = acc;
                }
            }
        });
    }
    return out;
}

py::array_t<double> pairwise_bray_curtis(Matrix X_, Matrix Y_) {
    const auto Xb = require_matrix(X_, "X");
    const auto Yb = require_matrix(Y_, "Y");
    require_same_features(Xb, Yb);
    const auto* X = static_cast<const double*>(Xb.ptr);
    const auto* Y = static_cast<const double*>(Yb.ptr);
    const py::ssize_t n = Xb.shape[0];
    const py::ssize_t m = Yb.shape[0];
    const py::ssize_t d = Xb.shape[1];
    constexpr double eps = 1e-12;

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
                    double num = 0.0;
                    double den = 0.0;
                    for (py::ssize_t k = 0; k < d; ++k) {
                        num += std::abs(xi[k] - yj[k]);
                        den += std::abs(xi[k] + yj[k]);
                    }
                    row[j] = num / std::max(den, eps);
                }
            }
        });
    }
    return out;
}

PYBIND11_MODULE(_distance_cpp, m) {
    m.doc() = "C++ accelerated pairwise distance functions";
    m.def("pairwise_squared_euclidean", &pairwise_squared_euclidean);
    m.def("pairwise_distances", &pairwise_distances);
    m.def("pairwise_manhattan_distances", &pairwise_manhattan);
    m.def("pairwise_chebyshev", &pairwise_chebyshev);
    m.def("pairwise_minkowski", &pairwise_minkowski);
    m.def("pairwise_canberra", &pairwise_canberra);
    m.def("pairwise_bray_curtis", &pairwise_bray_curtis);
}
