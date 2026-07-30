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

static py::array_t<double> peaks_output(
    py::object requested, py::ssize_t metrics, py::ssize_t joints) {
    if (requested.is_none()) {
        return py::array_t<double>(
            py::array::ShapeContainer(std::vector<py::ssize_t>{metrics, joints}));
    }
    auto output = py::array_t<double, py::array::c_style>::ensure(requested);
    if (!output) {
        throw std::invalid_argument("output must be a contiguous float64 NumPy array");
    }
    const auto info = output.request();
    if (info.ndim != 2 || info.shape[0] != metrics || info.shape[1] != joints) {
        throw std::invalid_argument("output must have shape (n_metrics, n_joints)");
    }
    return output;
}

static py::array_t<double> trajectory_peaks_batch(
    Matrix values_, py::object requested_output) {
    const auto values = values_.request();
    if (values.ndim != 3 || values.shape[0] < 1 || values.shape[1] < 1 || values.shape[2] < 1) {
        throw std::invalid_argument("values must have shape (n_samples, n_joints, n_metrics)");
    }
    const auto* value_data = static_cast<const double*>(values.ptr);
    require_finite(value_data, values.size, "values");
    const py::ssize_t samples = values.shape[0];
    const py::ssize_t joints = values.shape[1];
    const py::ssize_t metrics = values.shape[2];
    auto output = peaks_output(requested_output, metrics, joints);
    auto* output_data = static_cast<double*>(output.request().ptr);

    {
        py::gil_scoped_release release;
        for (py::ssize_t metric = 0; metric < metrics; ++metric) {
            for (py::ssize_t joint = 0; joint < joints; ++joint) {
                double maximum = 0.0;
                for (py::ssize_t sample = 0; sample < samples; ++sample) {
                    const double value = std::abs(
                        value_data[(sample * joints + joint) * metrics + metric]);
                    maximum = std::max(maximum, value);
                }
                output_data[metric * joints + joint] = maximum;
            }
        }
    }
    return output;
}

static py::array_t<double> trajectory_output(
    py::object requested,
    const std::vector<py::ssize_t>& shape,
    const char* name) {
    if (requested.is_none()) {
        return py::array_t<double>(py::array::ShapeContainer(shape));
    }
    auto output = py::array_t<double, py::array::c_style>::ensure(requested);
    if (!output) {
        throw std::invalid_argument(std::string(name) + " must be a contiguous float64 NumPy array");
    }
    const auto info = output.request();
    if (info.ndim != static_cast<py::ssize_t>(shape.size())) {
        throw std::invalid_argument(std::string(name) + " has the wrong number of dimensions");
    }
    for (std::size_t index = 0; index < shape.size(); ++index) {
        if (info.shape[index] != shape[index]) {
            throw std::invalid_argument(std::string(name) + " has the wrong shape");
        }
    }
    return output;
}

static py::tuple sample_joint_trajectory_segments(
    Matrix waypoints_, Vector durations_, py::ssize_t samples_per_segment,
    const std::string& kind, py::object requested_time, py::object requested_position,
    py::object requested_velocity, py::object requested_acceleration) {
    const auto waypoints = waypoints_.request();
    const auto durations = durations_.request();
    if (waypoints.ndim != 2 || waypoints.shape[0] < 2 || waypoints.shape[1] < 1) {
        throw std::invalid_argument("q_waypoints must have shape (n_waypoints, n_joints)");
    }
    if (durations.ndim != 1 || durations.shape[0] != waypoints.shape[0] - 1) {
        throw std::invalid_argument("durations must have one entry per segment");
    }
    if (samples_per_segment < 1) {
        throw std::invalid_argument("num_samples_per_segment must be at least 1");
    }
    if (kind != "cubic" && kind != "quintic") {
        throw std::invalid_argument("kind must be 'cubic' or 'quintic'");
    }
    const bool cubic = kind == "cubic";
    const py::ssize_t segments = waypoints.shape[0] - 1;
    const py::ssize_t joints = waypoints.shape[1];
    const py::ssize_t total_samples = samples_per_segment
        + (segments - 1) * (samples_per_segment - 1);
    const auto* waypoint_data = static_cast<const double*>(waypoints.ptr);
    const auto* duration_data = static_cast<const double*>(durations.ptr);
    require_finite(waypoint_data, waypoints.size, "q_waypoints");
    require_finite(duration_data, durations.size, "durations");
    for (py::ssize_t segment = 0; segment < segments; ++segment) {
        if (duration_data[segment] <= 0.0) {
            throw std::invalid_argument("durations must be positive");
        }
    }

    auto times = trajectory_output(
        requested_time, std::vector<py::ssize_t>{total_samples}, "time");
    const auto matrix_shape = std::vector<py::ssize_t>{total_samples, joints};
    auto positions = trajectory_output(requested_position, matrix_shape, "position");
    auto velocities = trajectory_output(requested_velocity, matrix_shape, "velocity");
    auto accelerations = trajectory_output(requested_acceleration, matrix_shape, "acceleration");
    auto* time_data = static_cast<double*>(times.request().ptr);
    auto* position_data = static_cast<double*>(positions.request().ptr);
    auto* velocity_data = static_cast<double*>(velocities.request().ptr);
    auto* acceleration_data = static_cast<double*>(accelerations.request().ptr);

    {
        py::gil_scoped_release release;
        py::ssize_t output_index = 0;
        double time_offset = 0.0;
        for (py::ssize_t segment = 0; segment < segments; ++segment) {
            const double duration = duration_data[segment];
            const double* start = waypoint_data + segment * joints;
            const double* end = start + joints;
            for (py::ssize_t sample = 0; sample < samples_per_segment; ++sample) {
                if (segment > 0 && sample == 0) {
                    continue;
                }
                const double tau = samples_per_segment == 1
                    ? 0.0
                    : static_cast<double>(sample) / static_cast<double>(samples_per_segment - 1);
                const double tau2 = tau * tau;
                const double tau3 = tau2 * tau;
                double s = 0.0;
                double ds = 0.0;
                double dds = 0.0;
                if (cubic) {
                    s = 3.0 * tau2 - 2.0 * tau3;
                    ds = (6.0 * tau - 6.0 * tau2) / duration;
                    dds = (6.0 - 12.0 * tau) / (duration * duration);
                } else {
                    const double tau4 = tau3 * tau;
                    const double tau5 = tau4 * tau;
                    s = 10.0 * tau3 - 15.0 * tau4 + 6.0 * tau5;
                    ds = (30.0 * tau2 - 60.0 * tau3 + 30.0 * tau4) / duration;
                    dds = (60.0 * tau - 180.0 * tau2 + 120.0 * tau3)
                        / (duration * duration);
                }
                time_data[output_index] = time_offset + tau * duration;
                for (py::ssize_t joint = 0; joint < joints; ++joint) {
                    const double delta = end[joint] - start[joint];
                    const py::ssize_t offset = output_index * joints + joint;
                    position_data[offset] = start[joint] + s * delta;
                    velocity_data[offset] = ds * delta;
                    acceleration_data[offset] = dds * delta;
                }
                ++output_index;
            }
            time_offset += duration;
        }
    }
    return py::make_tuple(times, positions, velocities, accelerations);
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
    m.def(
        "trajectory_peaks_batch", &trajectory_peaks_batch,
        py::arg("values"), py::arg("output") = py::none());
    m.def(
        "sample_joint_trajectory_segments", &sample_joint_trajectory_segments,
        py::arg("q_waypoints"), py::arg("durations"),
        py::arg("num_samples_per_segment") = 100, py::arg("kind") = "quintic",
        py::arg("time") = py::none(), py::arg("position") = py::none(),
        py::arg("velocity") = py::none(), py::arg("acceleration") = py::none());
}
