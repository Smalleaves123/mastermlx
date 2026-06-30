#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <algorithm>
#include <stdexcept>
namespace py = pybind11;

// ---------------------------------------------------------------------------
//  RBF (Gaussian) kernel  —  exp(-gamma * ||x - y||^2)
//  Uses the squared-Euclidean trick to avoid a 3D intermediate:
//    ||x-y||^2 = ||x||^2 + ||y||^2 - 2 x·y
//  Memory: O(n*m) instead of O(n*m*d + n*m)
// ---------------------------------------------------------------------------
py::array_t<double> rbf_kernel(py::array_t<double> X_, py::array_t<double> Y_, double gamma) {
    py::buffer_info Xb = X_.request(), Yb = Y_.request();
    double* X = (double*)Xb.ptr;
    double* Y = (double*)Yb.ptr;
    int n = (int)Xb.shape[0], m = (int)Yb.shape[0], d = (int)Xb.shape[1];

    // Pre-compute squared norms
    std::vector<double> xn2(n), yn2(m);
    for (int i = 0; i < n; i++) {
        double s = 0.0;
        for (int k = 0; k < d; k++) s += X[i * d + k] * X[i * d + k];
        xn2[i] = s;
    }
    for (int j = 0; j < m; j++) {
        double s = 0.0;
        for (int k = 0; k < d; k++) s += Y[j * d + k] * Y[j * d + k];
        yn2[j] = s;
    }

    py::array_t<double> out({n, m});
    py::buffer_info ob = out.request();
    double* ptr = (double*)ob.ptr;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            double dot = 0.0;
            for (int k = 0; k < d; k++)
                dot += X[i * d + k] * Y[j * d + k];
            double d2 = xn2[i] + yn2[j] - 2.0 * dot;
            ptr[i * m + j] = std::exp(-gamma * std::max(d2, 0.0));
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
//  RBF kernel (fast path) — uses pre-computed squared norms
// ---------------------------------------------------------------------------
py::array_t<double> rbf_kernel_fast(
    py::array_t<double> X_, py::array_t<double> Y_,
    py::array_t<double> xn2_, py::array_t<double> yn2_, double gamma)
{
    py::buffer_info Xb = X_.request(), Yb = Y_.request();
    py::buffer_info xn2b = xn2_.request(), yn2b = yn2_.request();
    double* X   = (double*)Xb.ptr;
    double* Y   = (double*)Yb.ptr;
    double* xn2 = (double*)xn2b.ptr;
    double* yn2 = (double*)yn2b.ptr;
    int n = (int)Xb.shape[0], m = (int)Yb.shape[0], d = (int)Xb.shape[1];

    py::array_t<double> out({n, m});
    py::buffer_info ob = out.request();
    double* ptr = (double*)ob.ptr;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            double dot = 0.0;
            for (int k = 0; k < d; k++)
                dot += X[i * d + k] * Y[j * d + k];
            double d2 = xn2[i] + yn2[j] - 2.0 * dot;
            ptr[i * m + j] = std::exp(-gamma * std::max(d2, 0.0));
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
//  Laplacian kernel  —  exp(-gamma * L1(x, y))
//  L1 distance computed inline, no 3D array.
// ---------------------------------------------------------------------------
py::array_t<double> laplacian_kernel(py::array_t<double> X_, py::array_t<double> Y_, double gamma) {
    py::buffer_info Xb = X_.request(), Yb = Y_.request();
    double* X = (double*)Xb.ptr;
    double* Y = (double*)Yb.ptr;
    int n = (int)Xb.shape[0], m = (int)Yb.shape[0], d = (int)Xb.shape[1];

    py::array_t<double> out({n, m});
    py::buffer_info ob = out.request();
    double* ptr = (double*)ob.ptr;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            double l1 = 0.0;
            for (int k = 0; k < d; k++)
                l1 += std::abs(X[i * d + k] - Y[j * d + k]);
            ptr[i * m + j] = std::exp(-gamma * l1);
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
//  Chi-squared kernel  —  exp( -gamma * 0.5 * sum( (x-y)^2 / (x+y) ) )
// ---------------------------------------------------------------------------
py::array_t<double> chi2_kernel(py::array_t<double> X_, py::array_t<double> Y_, double gamma) {
    py::buffer_info Xb = X_.request(), Yb = Y_.request();
    double* X = (double*)Xb.ptr;
    double* Y = (double*)Yb.ptr;
    int n = (int)Xb.shape[0], m = (int)Yb.shape[0], d = (int)Xb.shape[1];

    py::array_t<double> out({n, m});
    py::buffer_info ob = out.request();
    double* ptr = (double*)ob.ptr;
    const double eps = 1e-12;

    for (int i = 0; i < n * d; i++) {
        if (X[i] < 0.0) throw std::invalid_argument("chi2_kernel expects non-negative inputs");
    }
    for (int i = 0; i < m * d; i++) {
        if (Y[i] < 0.0) throw std::invalid_argument("chi2_kernel expects non-negative inputs");
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            double chi2 = 0.0;
            for (int k = 0; k < d; k++) {
                double num = X[i * d + k] - Y[j * d + k];
                num *= num;
                double den = X[i * d + k] + Y[j * d + k];
                chi2 += num / std::max(den, eps);
            }
            ptr[i * m + j] = std::exp(-gamma * 0.5 * chi2);
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
//  Additive chi-squared kernel  —  sum( 2*x*y / (x+y) )
// ---------------------------------------------------------------------------
py::array_t<double> additive_chi2_kernel(py::array_t<double> X_, py::array_t<double> Y_) {
    py::buffer_info Xb = X_.request(), Yb = Y_.request();
    double* X = (double*)Xb.ptr;
    double* Y = (double*)Yb.ptr;
    int n = (int)Xb.shape[0], m = (int)Yb.shape[0], d = (int)Xb.shape[1];

    py::array_t<double> out({n, m});
    py::buffer_info ob = out.request();
    double* ptr = (double*)ob.ptr;
    const double eps = 1e-12;

    for (int i = 0; i < n * d; i++) {
        if (X[i] < 0.0) throw std::invalid_argument("additive_chi2_kernel expects non-negative inputs");
    }
    for (int i = 0; i < m * d; i++) {
        if (Y[i] < 0.0) throw std::invalid_argument("additive_chi2_kernel expects non-negative inputs");
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            double acc = 0.0;
            for (int k = 0; k < d; k++) {
                double num = 2.0 * X[i * d + k] * Y[j * d + k];
                double den = X[i * d + k] + Y[j * d + k];
                acc += num / std::max(den, eps);
            }
            ptr[i * m + j] = acc;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
//  Hellinger kernel  —  sum( sqrt(x) * sqrt(y) )
// ---------------------------------------------------------------------------
py::array_t<double> hellinger_kernel(py::array_t<double> X_, py::array_t<double> Y_) {
    py::buffer_info Xb = X_.request(), Yb = Y_.request();
    double* X = (double*)Xb.ptr;
    double* Y = (double*)Yb.ptr;
    int n = (int)Xb.shape[0], m = (int)Yb.shape[0], d = (int)Xb.shape[1];

    py::array_t<double> out({n, m});
    py::buffer_info ob = out.request();
    double* ptr = (double*)ob.ptr;

    for (int i = 0; i < n * d; i++) {
        if (X[i] < 0.0) throw std::invalid_argument("hellinger_kernel expects non-negative inputs");
    }
    for (int i = 0; i < m * d; i++) {
        if (Y[i] < 0.0) throw std::invalid_argument("hellinger_kernel expects non-negative inputs");
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            double acc = 0.0;
            for (int k = 0; k < d; k++)
                acc += std::sqrt(X[i * d + k]) * std::sqrt(Y[j * d + k]);
            ptr[i * m + j] = acc;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
//  Module definition
// ---------------------------------------------------------------------------
PYBIND11_MODULE(_kernels_cpp, m) {
    m.doc() = "C++ accelerated kernel functions";
    m.def("rbf_kernel",            &rbf_kernel);
    m.def("rbf_kernel_fast",       &rbf_kernel_fast);
    m.def("laplacian_kernel",      &laplacian_kernel);
    m.def("chi2_kernel",           &chi2_kernel);
    m.def("additive_chi2_kernel",  &additive_chi2_kernel);
    m.def("hellinger_kernel",      &hellinger_kernel);
}
