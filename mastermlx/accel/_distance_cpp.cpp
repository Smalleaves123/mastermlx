#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <algorithm>
#include <vector>
namespace py = pybind11;

// ---------------------------------------------------------------------------
//  Pairwise squared Euclidean  —  O(n*m*d), no intermediate 3D array
// ---------------------------------------------------------------------------
py::array_t<double> pairwise_squared_euclidean(py::array_t<double> X_, py::array_t<double> Y_) {
    py::buffer_info Xb = X_.request(), Yb = Y_.request();
    double* X = (double*)Xb.ptr;
    double* Y = (double*)Yb.ptr;
    int n = (int)Xb.shape[0], m = (int)Yb.shape[0], d = (int)Xb.shape[1];
    py::array_t<double> out({n, m});
    py::buffer_info ob = out.request();
    double* ptr = (double*)ob.ptr;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            double acc = 0.0;
            for (int k = 0; k < d; k++) {
                double diff = X[i * d + k] - Y[j * d + k];
                acc += diff * diff;
            }
            ptr[i * m + j] = acc;
        }
    }
    return out;
}

py::array_t<double> pairwise_distances(py::array_t<double> X_, py::array_t<double> Y_) {
    py::array_t<double> sq = pairwise_squared_euclidean(X_, Y_);
    py::buffer_info sb = sq.request();
    double* ptr = (double*)sb.ptr;
    int n = (int)sb.shape[0], m = (int)sb.shape[1];
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            ptr[i * m + j] = std::sqrt(std::max(ptr[i * m + j], 0.0));
    return sq;
}

// ---------------------------------------------------------------------------
//  Pairwise Manhattan (L1)
// ---------------------------------------------------------------------------
py::array_t<double> pairwise_manhattan(py::array_t<double> X_, py::array_t<double> Y_) {
    py::buffer_info Xb = X_.request(), Yb = Y_.request();
    double* X = (double*)Xb.ptr;
    double* Y = (double*)Yb.ptr;
    int n = (int)Xb.shape[0], m = (int)Yb.shape[0], d = (int)Xb.shape[1];
    py::array_t<double> out({n, m});
    py::buffer_info ob = out.request();
    double* ptr = (double*)ob.ptr;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            double acc = 0.0;
            for (int k = 0; k < d; k++)
                acc += std::abs(X[i * d + k] - Y[j * d + k]);
            ptr[i * m + j] = acc;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
//  Pairwise Chebyshev  —  max(|x_k - y_k|) over k
// ---------------------------------------------------------------------------
py::array_t<double> pairwise_chebyshev(py::array_t<double> X_, py::array_t<double> Y_) {
    py::buffer_info Xb = X_.request(), Yb = Y_.request();
    double* X = (double*)Xb.ptr;
    double* Y = (double*)Yb.ptr;
    int n = (int)Xb.shape[0], m = (int)Yb.shape[0], d = (int)Xb.shape[1];
    py::array_t<double> out({n, m});
    py::buffer_info ob = out.request();
    double* ptr = (double*)ob.ptr;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            double best = 0.0;
            for (int k = 0; k < d; k++) {
                double diff = std::abs(X[i * d + k] - Y[j * d + k]);
                if (diff > best) best = diff;
            }
            ptr[i * m + j] = best;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
//  Pairwise Minkowski  —  ( sum(|x-y|^p) )^(1/p)
// ---------------------------------------------------------------------------
py::array_t<double> pairwise_minkowski(py::array_t<double> X_, py::array_t<double> Y_, double p) {
    py::buffer_info Xb = X_.request(), Yb = Y_.request();
    double* X = (double*)Xb.ptr;
    double* Y = (double*)Yb.ptr;
    int n = (int)Xb.shape[0], m = (int)Yb.shape[0], d = (int)Xb.shape[1];
    py::array_t<double> out({n, m});
    py::buffer_info ob = out.request();
    double* ptr = (double*)ob.ptr;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            double acc = 0.0;
            for (int k = 0; k < d; k++)
                acc += std::pow(std::abs(X[i * d + k] - Y[j * d + k]), p);
            ptr[i * m + j] = std::pow(acc, 1.0 / p);
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
//  Pairwise Canberra  —  sum( |x_k - y_k| / (|x_k| + |y_k|) )
// ---------------------------------------------------------------------------
py::array_t<double> pairwise_canberra(py::array_t<double> X_, py::array_t<double> Y_) {
    py::buffer_info Xb = X_.request(), Yb = Y_.request();
    double* X = (double*)Xb.ptr;
    double* Y = (double*)Yb.ptr;
    int n = (int)Xb.shape[0], m = (int)Yb.shape[0], d = (int)Xb.shape[1];
    py::array_t<double> out({n, m});
    py::buffer_info ob = out.request();
    double* ptr = (double*)ob.ptr;
    const double eps = 1e-12;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            double acc = 0.0;
            for (int k = 0; k < d; k++) {
                double num = std::abs(X[i * d + k] - Y[j * d + k]);
                double den = std::abs(X[i * d + k]) + std::abs(Y[j * d + k]);
                acc += num / std::max(den, eps);
            }
            ptr[i * m + j] = acc;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
//  Pairwise Bray-Curtis  —  sum(|x-y|) / sum(|x+y|)
// ---------------------------------------------------------------------------
py::array_t<double> pairwise_bray_curtis(py::array_t<double> X_, py::array_t<double> Y_) {
    py::buffer_info Xb = X_.request(), Yb = Y_.request();
    double* X = (double*)Xb.ptr;
    double* Y = (double*)Yb.ptr;
    int n = (int)Xb.shape[0], m = (int)Yb.shape[0], d = (int)Xb.shape[1];
    py::array_t<double> out({n, m});
    py::buffer_info ob = out.request();
    double* ptr = (double*)ob.ptr;
    const double eps = 1e-12;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            double num = 0.0, den = 0.0;
            for (int k = 0; k < d; k++) {
                num += std::abs(X[i * d + k] - Y[j * d + k]);
                den += std::abs(X[i * d + k] + Y[j * d + k]);
            }
            ptr[i * m + j] = num / std::max(den, eps);
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
//  Module definition
// ---------------------------------------------------------------------------
PYBIND11_MODULE(_distance_cpp, m) {
    m.doc() = "C++ accelerated pairwise distance functions";
    m.def("pairwise_squared_euclidean",   &pairwise_squared_euclidean);
    m.def("pairwise_distances",           &pairwise_distances);
    m.def("pairwise_manhattan_distances", &pairwise_manhattan);
    m.def("pairwise_chebyshev",           &pairwise_chebyshev);
    m.def("pairwise_minkowski",           &pairwise_minkowski);
    m.def("pairwise_canberra",            &pairwise_canberra);
    m.def("pairwise_bray_curtis",         &pairwise_bray_curtis);
}
