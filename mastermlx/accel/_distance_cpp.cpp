#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
namespace py = pybind11;

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

PYBIND11_MODULE(_distance_cpp, m) {
    m.doc() = "C++ accelerated pairwise distances";
    m.def("pairwise_squared_euclidean", &pairwise_squared_euclidean);
    m.def("pairwise_distances", &pairwise_distances);
    m.def("pairwise_manhattan_distances", &pairwise_manhattan);
}
