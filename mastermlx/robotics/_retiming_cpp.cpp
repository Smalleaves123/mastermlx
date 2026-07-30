#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

using Matrix = py::array_t<double, py::array::c_style | py::array::forcecast>;
using Vector = py::array_t<double, py::array::c_style | py::array::forcecast>;

static void require_finite(const double* values, py::ssize_t size, const char* name) {
    for (py::ssize_t index = 0; index < size; ++index) {
        if (!std::isfinite(values[index])) {
            throw std::invalid_argument(std::string(name) + " must contain only finite values");
        }
    }
}

static Vector optional_limits(py::handle value, py::ssize_t joints, const char* name, bool* present) {
    *present = !value.is_none();
    if (!*present) {
        return Vector();
    }
    Vector limits = Vector::ensure(value);
    if (!limits) {
        throw std::invalid_argument(std::string(name) + " must be array-like");
    }
    const auto info = limits.request();
    if (info.ndim != 1 || info.shape[0] != joints) {
        throw std::invalid_argument(std::string(name) + " must have shape (n_joints,)");
    }
    const auto* data = static_cast<const double*>(info.ptr);
    require_finite(data, joints, name);
    for (py::ssize_t index = 0; index < joints; ++index) {
        if (data[index] <= 0.0) {
            throw std::invalid_argument(std::string(name) + " must contain only positive values");
        }
    }
    return limits;
}

static py::tuple retime_quintic_path(
    Matrix path_,
    Vector velocity_limits_,
    py::object acceleration_limits,
    py::object jerk_limits,
    py::ssize_t num_samples_per_segment,
    double minimum_duration) {
    const auto path = path_.request();
    const auto velocity_limits = velocity_limits_.request();
    if (path.ndim != 2 || path.shape[0] < 2 || path.shape[1] < 1) {
        throw std::invalid_argument("path must have shape (n_points, n_joints) with at least two points");
    }
    if (velocity_limits.ndim != 1 || velocity_limits.shape[0] != path.shape[1]) {
        throw std::invalid_argument("velocity_limits must have shape (n_joints,)");
    }
    if (num_samples_per_segment < 2 || !std::isfinite(minimum_duration) || minimum_duration <= 0.0) {
        throw std::invalid_argument("num_samples_per_segment must be at least 2 and minimum_duration must be positive");
    }

    const py::ssize_t segments = path.shape[0] - 1;
    const py::ssize_t joints = path.shape[1];
    const auto* path_data = static_cast<const double*>(path.ptr);
    const auto* velocity_data = static_cast<const double*>(velocity_limits.ptr);
    require_finite(path_data, path.shape[0] * joints, "path");
    require_finite(velocity_data, joints, "velocity_limits");
    for (py::ssize_t index = 0; index < joints; ++index) {
        if (velocity_data[index] <= 0.0) {
            throw std::invalid_argument("velocity_limits must contain only positive values");
        }
    }

    bool has_acceleration = false;
    bool has_jerk = false;
    Vector acceleration = optional_limits(acceleration_limits, joints, "acceleration_limits", &has_acceleration);
    Vector jerk = optional_limits(jerk_limits, joints, "jerk_limits", &has_jerk);
    const auto* acceleration_data = has_acceleration
        ? static_cast<const double*>(acceleration.request().ptr)
        : nullptr;
    const auto* jerk_data = has_jerk ? static_cast<const double*>(jerk.request().ptr) : nullptr;

    std::vector<double> durations(static_cast<std::size_t>(segments), minimum_duration);
    constexpr double max_velocity = 1.875;
    const double max_acceleration = 10.0 / std::sqrt(3.0);
    constexpr double max_jerk = 60.0;
    for (py::ssize_t segment = 0; segment < segments; ++segment) {
        double duration = minimum_duration;
        for (py::ssize_t joint = 0; joint < joints; ++joint) {
            const double delta = std::abs(
                path_data[(segment + 1) * joints + joint] - path_data[segment * joints + joint]);
            duration = std::max(duration, max_velocity * delta / velocity_data[joint]);
            if (has_acceleration) {
                duration = std::max(duration, std::sqrt(max_acceleration * delta / acceleration_data[joint]));
            }
            if (has_jerk) {
                duration = std::max(duration, std::cbrt(max_jerk * delta / jerk_data[joint]));
            }
        }
        durations[static_cast<std::size_t>(segment)] = duration;
    }

    const py::ssize_t total_samples = num_samples_per_segment
        + (segments - 1) * (num_samples_per_segment - 1);
    py::array_t<double> times(py::array::ShapeContainer(std::vector<py::ssize_t>{total_samples}));
    py::array_t<double> positions(
        py::array::ShapeContainer(std::vector<py::ssize_t>{total_samples, joints}));
    py::array_t<double> velocities(
        py::array::ShapeContainer(std::vector<py::ssize_t>{total_samples, joints}));
    py::array_t<double> accelerations_out(
        py::array::ShapeContainer(std::vector<py::ssize_t>{total_samples, joints}));
    py::array_t<double> jerks(
        py::array::ShapeContainer(std::vector<py::ssize_t>{total_samples, joints}));
    py::array_t<double> durations_out(
        py::array::ShapeContainer(std::vector<py::ssize_t>{segments}));

    auto* time_data = static_cast<double*>(times.request().ptr);
    auto* position_data = static_cast<double*>(positions.request().ptr);
    auto* velocity_output = static_cast<double*>(velocities.request().ptr);
    auto* acceleration_output = static_cast<double*>(accelerations_out.request().ptr);
    auto* jerk_output = static_cast<double*>(jerks.request().ptr);
    auto* duration_output = static_cast<double*>(durations_out.request().ptr);
    for (py::ssize_t segment = 0; segment < segments; ++segment) {
        duration_output[segment] = durations[static_cast<std::size_t>(segment)];
    }

    {
        py::gil_scoped_release release;
        py::ssize_t output_index = 0;
        double time_offset = 0.0;
        for (py::ssize_t segment = 0; segment < segments; ++segment) {
            const double duration = durations[static_cast<std::size_t>(segment)];
            for (py::ssize_t sample = 0; sample < num_samples_per_segment; ++sample) {
                if (segment > 0 && sample == 0) {
                    continue;
                }
                const double tau = static_cast<double>(sample)
                    / static_cast<double>(num_samples_per_segment - 1);
                const double tau2 = tau * tau;
                const double tau3 = tau2 * tau;
                const double tau4 = tau3 * tau;
                const double tau5 = tau4 * tau;
                const double s = 10.0 * tau3 - 15.0 * tau4 + 6.0 * tau5;
                const double ds = (30.0 * tau2 - 60.0 * tau3 + 30.0 * tau4) / duration;
                const double dds = (60.0 * tau - 180.0 * tau2 + 120.0 * tau3) / (duration * duration);
                const double jerk_scale = (60.0 - 360.0 * tau + 360.0 * tau2)
                    / (duration * duration * duration);
                time_data[output_index] = time_offset + tau * duration;
                for (py::ssize_t joint = 0; joint < joints; ++joint) {
                    const double delta = path_data[(segment + 1) * joints + joint] - path_data[segment * joints + joint];
                    const py::ssize_t output_offset = output_index * joints + joint;
                    position_data[output_offset] = path_data[segment * joints + joint] + s * delta;
                    velocity_output[output_offset] = ds * delta;
                    acceleration_output[output_offset] = dds * delta;
                    jerk_output[output_offset] = jerk_scale * delta;
                }
                ++output_index;
            }
            time_offset += duration;
        }
    }
    return py::make_tuple(times, positions, velocities, accelerations_out, jerks, durations_out);
}

PYBIND11_MODULE(_retiming_cpp, m) {
    m.doc() = "C++ accelerated quintic trajectory retiming";
    m.def(
        "retime_quintic_path", &retime_quintic_path,
        py::arg("path"), py::arg("velocity_limits"),
        py::arg("acceleration_limits") = py::none(), py::arg("jerk_limits") = py::none(),
        py::arg("num_samples_per_segment") = 101, py::arg("minimum_duration") = 1e-3);
}
