#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cmath>
#include <algorithm>
#include <vector>
namespace py = pybind11;

std::pair<py::array_t<int>, double> dtw_path(
    py::array_t<double> x_, py::array_t<double> y_, int window)
{
    auto x = x_.unchecked<1>(), y = y_.unchecked<1>();
    int n = x.shape(0), m = y.shape(0);
    if (window < std::abs(n - m)) window = std::abs(n - m);

    std::vector<double> dp((n + 1) * (m + 1), 1e308);
    dp[0] = 0.0;

    for (int i = 1; i <= n; i++) {
        int lo = std::max(1, i - window), hi = std::min(m, i + window);
        for (int j = lo; j <= hi; j++) {
            double cost = std::abs(x(i - 1) - y(j - 1));
            double d1 = dp[(i - 1) * (m + 1) + j];
            double d2 = dp[i * (m + 1) + (j - 1)];
            double d3 = dp[(i - 1) * (m + 1) + (j - 1)];
            dp[i * (m + 1) + j] = cost + std::min({d1, d2, d3});
        }
    }

    std::vector<int> path_i, path_j;
    int pi = n, pj = m;
    while (pi > 0 && pj > 0) {
        path_i.push_back(pi - 1); path_j.push_back(pj - 1);
        double d1 = dp[(pi-1)*(m+1) + pj], d2 = dp[pi*(m+1) + (pj-1)], d3 = dp[(pi-1)*(m+1) + (pj-1)];
        if (d1 <= d2 && d1 <= d3) pi--;
        else if (d2 <= d1 && d2 <= d3) pj--;
        else { pi--; pj--; }
    }

    int k = path_i.size();
    auto path_arr = py::array_t<int>({k, 2});
    auto p = path_arr.mutable_unchecked<2>();
    for (int t = 0; t < k; t++) { p(k-1-t, 0) = path_i[t]; p(k-1-t, 1) = path_j[t]; }
    return {path_arr, dp[n * (m + 1) + m]};
}

PYBIND11_MODULE(_dtw, m) {
    m.doc() = "C++ accelerated DTW";
    m.def("dtw_path", &dtw_path);
}
