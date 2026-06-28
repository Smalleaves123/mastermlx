#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
namespace py = pybind11;

py::array_t<double> pairwise_squared_euclidean(py::array_t<double> X_, py::array_t<double> Y_) {
    auto X = X_.unchecked<2>(), Y = Y_.unchecked<2>();
    ssize_t n = X.shape(0), m = Y.shape(0), d = X.shape(1);
    auto out = py::array_t<double>({n, m});
    auto ptr = out.mutable_unchecked<2>();
    for (ssize_t i = 0; i < n; i++)
        for (ssize_t j = 0; j < m; j++) {
            double acc = 0.0;
            for (ssize_t k = 0; k < d; k++) {
                double diff = X(i, k) - Y(j, k);
                acc += diff * diff;
            }
            ptr(i, j) = acc;
        }
    return out;
}

py::array_t<double> pairwise_distances(py::array_t<double> X_, py::array_t<double> Y_) {
    auto sq = pairwise_squared_euclidean(X_, Y_);
    auto r = sq.mutable_unchecked<2>();
    for (ssize_t i = 0; i < r.shape(0); i++)
        for (ssize_t j = 0; j < r.shape(1); j++)
            r(i, j) = std::sqrt(std::max(r(i, j), 0.0));
    return sq;
}

py::array_t<double> pairwise_manhattan(py::array_t<double> X_, py::array_t<double> Y_) {
    auto X = X_.unchecked<2>(), Y = Y_.unchecked<2>();
    ssize_t n = X.shape(0), m = Y.shape(0), d = X.shape(1);
    auto out = py::array_t<double>({n, m});
    auto ptr = out.mutable_unchecked<2>();
    for (ssize_t i = 0; i < n; i++)
        for (ssize_t j = 0; j < m; j++) {
            double acc = 0.0;
            for (ssize_t k = 0; k < d; k++)
                acc += std::abs(X(i, k) - Y(j, k));
            ptr(i, j) = acc;
        }
    return out;
}

PYBIND11_MODULE(_distance_cpp, m) {
    m.doc() = "C++ accelerated pairwise distances";
    m.def("pairwise_squared_euclidean", &pairwise_squared_euclidean);
    m.def("pairwise_distances", &pairwise_distances);
    m.def("pairwise_manhattan_distances", &pairwise_manhattan);
}
