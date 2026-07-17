#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

using Series = py::array_t<double, py::array::c_style | py::array::forcecast>;

static py::buffer_info require_series(const Series& value, const char* name) {
    auto info = value.request();
    if (info.ndim != 1) {
        throw std::invalid_argument(std::string(name) + " must be a 1D array");
    }
    if (info.shape[0] <= 0) {
        throw std::invalid_argument(std::string(name) + " must be non-empty");
    }
    const auto* data = static_cast<const double*>(info.ptr);
    for (py::ssize_t index = 0; index < info.shape[0]; ++index) {
        if (!std::isfinite(data[index])) {
            throw std::invalid_argument(std::string(name) + " must contain only finite values");
        }
    }
    return info;
}

static py::ssize_t normalize_window(py::ssize_t window, py::ssize_t n, py::ssize_t m) {
    if (window < 0) {
        throw std::invalid_argument("window must be non-negative");
    }
    return std::max(window, std::abs(n - m));
}

struct Band {
    std::vector<py::ssize_t> lo;
    std::vector<py::ssize_t> hi;
    std::vector<std::size_t> offset;
    std::size_t size;
};

static Band make_band(py::ssize_t n, py::ssize_t m, py::ssize_t window) {
    if (n == std::numeric_limits<py::ssize_t>::max()) {
        throw std::overflow_error("DTW input is too large");
    }
    Band band{
        std::vector<py::ssize_t>(static_cast<std::size_t>(n + 1)),
        std::vector<py::ssize_t>(static_cast<std::size_t>(n + 1)),
        std::vector<std::size_t>(static_cast<std::size_t>(n + 1)),
        0,
    };
    for (py::ssize_t i = 0; i <= n; ++i) {
        const py::ssize_t lo = i == 0 ? 0 : std::max<py::ssize_t>(1, i - window);
        const py::ssize_t hi = i == 0 ? 0
            : (window >= m - i ? m : i + window);
        const auto width = static_cast<std::size_t>(hi - lo + 1);
        const auto max_size = std::vector<double>().max_size();
        if (band.size > max_size || width > max_size - band.size) {
            throw std::overflow_error("DTW band is too large");
        }
        band.lo[static_cast<std::size_t>(i)] = lo;
        band.hi[static_cast<std::size_t>(i)] = hi;
        band.offset[static_cast<std::size_t>(i)] = band.size;
        band.size += width;
    }
    return band;
}

static double band_value(
    const std::vector<double>& values, const Band& band, py::ssize_t i, py::ssize_t j) {
    if (i < 0 || j < 0 || i >= static_cast<py::ssize_t>(band.lo.size())) {
        return std::numeric_limits<double>::infinity();
    }
    const auto row = static_cast<std::size_t>(i);
    if (j < band.lo[row] || j > band.hi[row]) {
        return std::numeric_limits<double>::infinity();
    }
    const auto index = band.offset[row] + static_cast<std::size_t>(j - band.lo[row]);
    return values[index];
}

std::pair<py::array_t<int>, double> dtw_path(
    Series x_, Series y_, py::ssize_t window) {
    const auto xb = require_series(x_, "x");
    const auto yb = require_series(y_, "y");
    const auto* x = static_cast<const double*>(xb.ptr);
    const auto* y = static_cast<const double*>(yb.ptr);
    const py::ssize_t n = xb.shape[0];
    const py::ssize_t m = yb.shape[0];
    if (n > std::numeric_limits<int>::max() || m > std::numeric_limits<int>::max()) {
        throw std::overflow_error("DTW path indices exceed int range");
    }
    window = normalize_window(window, n, m);

    const Band band = make_band(n, m, window);
    std::vector<double> dp(band.size, std::numeric_limits<double>::infinity());
    std::vector<int> path_i;
    std::vector<int> path_j;
    {
        py::gil_scoped_release release;
        dp[band.offset[0]] = 0.0;
        for (py::ssize_t i = 1; i <= n; ++i) {
            const auto row = static_cast<std::size_t>(i);
            for (py::ssize_t j = band.lo[row]; j <= band.hi[row]; ++j) {
                const double cost = std::abs(x[i - 1] - y[j - 1]);
                const auto current = band.offset[row]
                    + static_cast<std::size_t>(j - band.lo[row]);
                const double up = band_value(dp, band, i - 1, j);
                const double left = band_value(dp, band, i, j - 1);
                const double diagonal = band_value(dp, band, i - 1, j - 1);
                dp[current] = cost + std::min(up, std::min(left, diagonal));
            }
        }

        py::ssize_t pi = n;
        py::ssize_t pj = m;
        while (pi > 0 && pj > 0) {
            path_i.push_back(static_cast<int>(pi - 1));
            path_j.push_back(static_cast<int>(pj - 1));
            const double up = band_value(dp, band, pi - 1, pj);
            const double left = band_value(dp, band, pi, pj - 1);
            const double diagonal = band_value(dp, band, pi - 1, pj - 1);
            if (up <= left && up <= diagonal) {
                --pi;
            } else if (left <= up && left <= diagonal) {
                --pj;
            } else {
                --pi;
                --pj;
            }
        }
    }

    const py::ssize_t length = static_cast<py::ssize_t>(path_i.size());
    py::array_t<int> path_arr({length, static_cast<py::ssize_t>(2)});
    auto* output = static_cast<int*>(path_arr.request().ptr);
    for (py::ssize_t index = 0; index < length; ++index) {
        const py::ssize_t source = length - 1 - index;
        output[index * 2] = path_i[static_cast<std::size_t>(source)];
        output[index * 2 + 1] = path_j[static_cast<std::size_t>(source)];
    }
    return {path_arr, band_value(dp, band, n, m)};
}

double dtw_distance(Series x_, Series y_, py::ssize_t window) {
    const auto xb = require_series(x_, "x");
    const auto yb = require_series(y_, "y");
    const auto* x = static_cast<const double*>(xb.ptr);
    const auto* y = static_cast<const double*>(yb.ptr);
    const py::ssize_t n = xb.shape[0];
    const py::ssize_t m = yb.shape[0];
    window = normalize_window(window, n, m);

    const auto max_ssize = std::numeric_limits<py::ssize_t>::max();
    const py::ssize_t width = window > (max_ssize - 1) / 2
        ? m
        : std::min<py::ssize_t>(m, 2 * window + 1);
    std::vector<double> previous(static_cast<std::size_t>(width),
                                 std::numeric_limits<double>::infinity());
    std::vector<double> current(static_cast<std::size_t>(width),
                                std::numeric_limits<double>::infinity());
    previous[0] = 0.0;
    py::ssize_t previous_lo = 0;
    py::ssize_t previous_hi = 0;
    {
        py::gil_scoped_release release;
        for (py::ssize_t i = 1; i <= n; ++i) {
            const py::ssize_t lo = std::max<py::ssize_t>(1, i - window);
            const py::ssize_t hi = window >= m - i ? m : i + window;
            const auto row_width = static_cast<std::size_t>(hi - lo + 1);
            std::fill(current.begin(), current.begin() + row_width,
                      std::numeric_limits<double>::infinity());
            for (py::ssize_t j = lo; j <= hi; ++j) {
                const double cost = std::abs(x[i - 1] - y[j - 1]);
                const double up = j >= previous_lo && j <= previous_hi
                    ? previous[static_cast<std::size_t>(j - previous_lo)]
                    : std::numeric_limits<double>::infinity();
                const double left = j - 1 >= lo && j - 1 <= hi
                    ? current[static_cast<std::size_t>(j - 1 - lo)]
                    : std::numeric_limits<double>::infinity();
                const double diagonal = j - 1 >= previous_lo && j - 1 <= previous_hi
                    ? previous[static_cast<std::size_t>(j - 1 - previous_lo)]
                    : std::numeric_limits<double>::infinity();
                current[static_cast<std::size_t>(j - lo)] =
                    cost + std::min(up, std::min(left, diagonal));
            }
            previous.swap(current);
            previous_lo = lo;
            previous_hi = hi;
        }
    }
    return previous[static_cast<std::size_t>(m - previous_lo)];
}

PYBIND11_MODULE(_dtw, m) {
    m.doc() = "C++ accelerated DTW";
    m.def("dtw_path", &dtw_path, py::arg("x"), py::arg("y"), py::arg("window"));
    m.def("dtw_distance", &dtw_distance,
          py::arg("x"), py::arg("y"), py::arg("window"));
}
