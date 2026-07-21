#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

using Vector = py::array_t<double, py::array::c_style | py::array::forcecast>;

static py::buffer_info require_vector(const Vector& value, const char* name) {
    auto info = value.request();
    if (info.ndim != 1 || info.shape[0] <= 0) {
        throw std::invalid_argument(std::string(name) + " must be a non-empty 1D array");
    }
    const auto* data = static_cast<const double*>(info.ptr);
    for (py::ssize_t i = 0; i < info.shape[0]; ++i) {
        if (!std::isfinite(data[i])) {
            throw std::invalid_argument(std::string(name) + " must contain only finite values");
        }
    }
    return info;
}

static py::buffer_info require_optional_vector(const Vector& value, const char* name) {
    auto info = value.request();
    if (info.ndim != 1) {
        throw std::invalid_argument(std::string(name) + " must be a 1D array");
    }
    const auto* data = static_cast<const double*>(info.ptr);
    for (py::ssize_t i = 0; i < info.shape[0]; ++i) {
        if (!std::isfinite(data[i])) {
            throw std::invalid_argument(std::string(name) + " must contain only finite values");
        }
    }
    return info;
}

using Matrix = py::array_t<double, py::array::c_style | py::array::forcecast>;

static py::buffer_info require_matrix(const Matrix& value, const char* name) {
    auto info = value.request();
    if (info.ndim != 2 || info.shape[0] <= 0 || info.shape[1] <= 0) {
        throw std::invalid_argument(std::string(name) + " must be a non-empty 2D array");
    }
    const auto* data = static_cast<const double*>(info.ptr);
    for (py::ssize_t i = 0; i < info.shape[0] * info.shape[1]; ++i) {
        if (!std::isfinite(data[i])) {
            throw std::invalid_argument(std::string(name) + " must contain only finite values");
        }
    }
    return info;
}

py::array_t<double> frame_signal(Vector x_, int frame_length, int hop_length, bool pad_end) {
    const auto xb = require_vector(x_, "x");
    if (frame_length < 1 || hop_length < 1) {
        throw std::invalid_argument("frame_length and hop_length must be at least 1");
    }

    const auto* x = static_cast<const double*>(xb.ptr);
    const py::ssize_t n = xb.shape[0];
    py::ssize_t pad = 0;
    if (pad_end) {
        if (n < frame_length) {
            pad = frame_length - n;
        } else {
            const py::ssize_t remainder = (n - frame_length) % hop_length;
            pad = remainder == 0 ? 0 : hop_length - remainder;
        }
    }
    const py::ssize_t total = n + pad;
    if (total < frame_length) {
        return py::array_t<double>(std::vector<py::ssize_t>{0, static_cast<py::ssize_t>(frame_length)});
    }

    const py::ssize_t n_frames = 1 + (total - frame_length) / hop_length;
    py::array_t<double> out(
        std::vector<py::ssize_t>{n_frames, static_cast<py::ssize_t>(frame_length)});
    auto* result = static_cast<double*>(out.request().ptr);
    {
        py::gil_scoped_release release;
        for (py::ssize_t i = 0; i < n_frames; ++i) {
            const py::ssize_t start = i * hop_length;
            for (py::ssize_t j = 0; j < frame_length; ++j) {
                const py::ssize_t index = start + j;
                result[i * frame_length + j] = index < n ? x[index] : 0.0;
            }
        }
    }
    return out;
}

py::array_t<double> iir_filter_1d(Vector x_, Vector b_, Vector a_) {
    const auto xb = require_vector(x_, "x");
    const auto bb = require_vector(b_, "b");
    const auto ab = require_vector(a_, "a");
    const auto* x = static_cast<const double*>(xb.ptr);
    const auto* b = static_cast<const double*>(bb.ptr);
    const auto* a = static_cast<const double*>(ab.ptr);
    if (a[0] == 0.0) {
        throw std::invalid_argument("a[0] must be non-zero");
    }

    const py::ssize_t n = xb.shape[0];
    const py::ssize_t nb = bb.shape[0];
    const py::ssize_t na = ab.shape[0];
    py::array_t<double> out(n);
    auto* y = static_cast<double*>(out.request().ptr);
    {
        py::gil_scoped_release release;
        for (py::ssize_t i = 0; i < n; ++i) {
            double value = 0.0;
            const py::ssize_t upper_b = std::min(nb, i + 1);
            for (py::ssize_t k = 0; k < upper_b; ++k) {
                value += b[k] * x[i - k];
            }
            const py::ssize_t upper_a = std::min(na, i + 1);
            for (py::ssize_t k = 1; k < upper_a; ++k) {
                value -= a[k] * y[i - k];
            }
            y[i] = value;
        }
    }
    return out;
}

py::array_t<py::ssize_t> ridge_path(Matrix score_, double smoothness, int max_jump) {
    const auto score_info = require_matrix(score_, "score");
    if (smoothness < 0.0 || !std::isfinite(smoothness)) {
        throw std::invalid_argument("smoothness must be non-negative and finite");
    }
    if (max_jump < -1) {
        throw std::invalid_argument("max_jump must be non-negative or -1");
    }

    const auto* score = static_cast<const double*>(score_info.ptr);
    const py::ssize_t n_freqs = score_info.shape[0];
    const py::ssize_t n_times = score_info.shape[1];
    const py::ssize_t size = n_freqs * n_times;
    py::array_t<double> dynamic(std::vector<py::ssize_t>{n_freqs, n_times});
    py::array_t<py::ssize_t> back(std::vector<py::ssize_t>{n_freqs, n_times});
    py::array_t<py::ssize_t> indices(n_times);
    auto* values = static_cast<double*>(dynamic.request().ptr);
    auto* previous = static_cast<py::ssize_t*>(back.request().ptr);
    auto* output = static_cast<py::ssize_t*>(indices.request().ptr);
    std::fill(values, values + size, -std::numeric_limits<double>::infinity());

    {
        py::gil_scoped_release release;
        for (py::ssize_t current = 0; current < n_freqs; ++current) {
            values[current * n_times] = score[current * n_times];
        }
        for (py::ssize_t t = 1; t < n_times; ++t) {
            for (py::ssize_t current = 0; current < n_freqs; ++current) {
                const py::ssize_t low = max_jump < 0 ? 0 : std::max<py::ssize_t>(0, current - max_jump);
                const py::ssize_t high = max_jump < 0
                    ? n_freqs
                    : std::min<py::ssize_t>(n_freqs, current + max_jump + 1);
                py::ssize_t best = low;
                double best_value = -std::numeric_limits<double>::infinity();
                for (py::ssize_t prior = low; prior < high; ++prior) {
                    const double delta = static_cast<double>(prior - current);
                    const double candidate = values[prior * n_times + t - 1] - smoothness * delta * delta;
                    if (candidate > best_value) {
                        best_value = candidate;
                        best = prior;
                    }
                }
                previous[current * n_times + t] = best;
                values[current * n_times + t] = score[current * n_times + t] + best_value;
            }
        }

        py::ssize_t best = 0;
        double best_value = values[n_times - 1];
        for (py::ssize_t current = 1; current < n_freqs; ++current) {
            const double value = values[current * n_times + n_times - 1];
            if (value > best_value) {
                best_value = value;
                best = current;
            }
        }
        output[n_times - 1] = best;
        for (py::ssize_t t = n_times - 1; t > 0; --t) {
            output[t - 1] = previous[output[t] * n_times + t];
        }
    }
    return indices;
}

py::tuple online_cusum(
    Vector x_,
    Vector baseline_,
    double baseline_mean,
    double positive,
    double negative,
    long long samples_seen,
    double threshold,
    double drift,
    int baseline_window,
    int cooldown_left,
    int cooldown) {
    const auto xb = require_vector(x_, "x");
    const auto baseline_info = require_optional_vector(baseline_, "baseline");
    if (threshold <= 0.0 || !std::isfinite(threshold)) {
        throw std::invalid_argument("threshold must be positive and finite");
    }
    if (!std::isfinite(drift)) {
        throw std::invalid_argument("drift must be finite");
    }
    if (baseline_window < 1) {
        throw std::invalid_argument("baseline_window must be at least 1");
    }
    if (cooldown_left < 0 || cooldown < 0) {
        throw std::invalid_argument("cooldown values must be non-negative");
    }

    const auto* x = static_cast<const double*>(xb.ptr);
    const auto* baseline = static_cast<const double*>(baseline_info.ptr);
    std::vector<double> base(baseline, baseline + baseline_info.shape[0]);
    std::vector<long long> events;
    const py::ssize_t n = xb.shape[0];
    for (py::ssize_t i = 0; i < n; ++i) {
        const long long index = samples_seen++;
        const double value = x[i];
        if (std::isnan(baseline_mean)) {
            base.push_back(value);
            if (static_cast<int>(base.size()) >= baseline_window) {
                double sum = 0.0;
                for (double item : base) {
                    sum += item;
                }
                baseline_mean = sum / static_cast<double>(base.size());
            }
            continue;
        }

        const double deviation = value - baseline_mean;
        positive = std::max(0.0, positive + deviation - drift);
        negative = std::min(0.0, negative + deviation + drift);
        if (cooldown_left > 0) {
            --cooldown_left;
            continue;
        }
        if (positive >= threshold || -negative >= threshold) {
            events.push_back(index);
            positive = 0.0;
            negative = 0.0;
            cooldown_left = cooldown;
        }
    }

    py::array_t<long long> event_array(events.size());
    auto* event_ptr = static_cast<long long*>(event_array.request().ptr);
    std::copy(events.begin(), events.end(), event_ptr);
    py::array_t<double> baseline_array(base.size());
    auto* baseline_ptr = static_cast<double*>(baseline_array.request().ptr);
    std::copy(base.begin(), base.end(), baseline_ptr);
    return py::make_tuple(
        event_array,
        baseline_array,
        baseline_mean,
        positive,
        negative,
        samples_seen,
        cooldown_left);
}

PYBIND11_MODULE(_signal_cpp, m) {
    m.doc() = "C++ accelerated signal-processing kernels";
    m.def("frame_signal", &frame_signal);
    m.def("iir_filter_1d", &iir_filter_1d);
    m.def("online_cusum", &online_cusum);
    m.def("ridge_path", &ridge_path);
}
