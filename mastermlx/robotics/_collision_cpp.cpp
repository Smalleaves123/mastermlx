#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace py = pybind11;

using Points = py::array_t<double, py::array::c_style | py::array::forcecast>;
using Parameters = py::array_t<double, py::array::c_style | py::array::forcecast>;
using Types = py::array_t<std::int8_t, py::array::c_style | py::array::forcecast>;

static inline double clamp_value(double value, double lower, double upper) {
    return std::max(lower, std::min(value, upper));
}

static inline double dot(const double* first, const double* second, py::ssize_t dims) {
    double result = 0.0;
    for (py::ssize_t index = 0; index < dims; ++index) {
        result += first[index] * second[index];
    }
    return result;
}

static inline double norm(const double* value, py::ssize_t dims) {
    return std::sqrt(dot(value, value, dims));
}

static double point_segment_distance(
    const double* point, const double* start, const double* end, py::ssize_t dims) {
    double edge[3] = {0.0, 0.0, 0.0};
    double relative[3] = {0.0, 0.0, 0.0};
    for (py::ssize_t index = 0; index < dims; ++index) {
        edge[index] = end[index] - start[index];
        relative[index] = point[index] - start[index];
    }
    const double length_sq = dot(edge, edge, dims);
    if (length_sq == 0.0) {
        return norm(relative, dims);
    }
    const double alpha = clamp_value(dot(relative, edge, dims) / length_sq, 0.0, 1.0);
    double difference[3] = {0.0, 0.0, 0.0};
    for (py::ssize_t index = 0; index < dims; ++index) {
        difference[index] = relative[index] - alpha * edge[index];
    }
    return norm(difference, dims);
}

static double segment_distance(
    const double* first_start,
    const double* first_end,
    const double* second_start,
    const double* second_end,
    py::ssize_t dims) {
    double first_edge[3] = {0.0, 0.0, 0.0};
    double second_edge[3] = {0.0, 0.0, 0.0};
    double relative[3] = {0.0, 0.0, 0.0};
    for (py::ssize_t index = 0; index < dims; ++index) {
        first_edge[index] = first_end[index] - first_start[index];
        second_edge[index] = second_end[index] - second_start[index];
        relative[index] = first_start[index] - second_start[index];
    }
    const double a = dot(first_edge, first_edge, dims);
    const double e = dot(second_edge, second_edge, dims);
    const double f = dot(second_edge, relative, dims);
    const double eps = 1e-12;
    double s = 0.0;
    double t = 0.0;
    if (a <= eps && e <= eps) {
        double difference[3] = {0.0, 0.0, 0.0};
        for (py::ssize_t index = 0; index < dims; ++index) {
            difference[index] = relative[index];
        }
        return norm(difference, dims);
    }
    if (a <= eps) {
        t = clamp_value(f / e, 0.0, 1.0);
    } else {
        const double c = dot(first_edge, relative, dims);
        if (e <= eps) {
            s = clamp_value(-c / a, 0.0, 1.0);
        } else {
            const double b = dot(first_edge, second_edge, dims);
            const double denominator = a * e - b * b;
            s = denominator <= eps
                ? 0.0
                : clamp_value((b * f - c * e) / denominator, 0.0, 1.0);
            t = (b * s + f) / e;
            if (t < 0.0) {
                t = 0.0;
                s = clamp_value(-c / a, 0.0, 1.0);
            } else if (t > 1.0) {
                t = 1.0;
                s = clamp_value((b - c) / a, 0.0, 1.0);
            }
        }
    }
    double difference[3] = {0.0, 0.0, 0.0};
    for (py::ssize_t index = 0; index < dims; ++index) {
        difference[index] = relative[index] + s * first_edge[index] - t * second_edge[index];
    }
    return norm(difference, dims);
}

static double point_box_clearance(
    const double* point, const double* lower, const double* upper, py::ssize_t dims) {
    double outside[3] = {0.0, 0.0, 0.0};
    bool inside = true;
    double nearest_face = std::numeric_limits<double>::infinity();
    for (py::ssize_t index = 0; index < dims; ++index) {
        outside[index] = std::max(std::max(lower[index] - point[index], point[index] - upper[index]), 0.0);
        inside = inside && point[index] >= lower[index] && point[index] <= upper[index];
        nearest_face = std::min(nearest_face, std::min(point[index] - lower[index], upper[index] - point[index]));
    }
    if (!inside) {
        return norm(outside, dims);
    }
    return -nearest_face;
}

static double point_clearance(
    const double* point,
    const double* params,
    std::int8_t type,
    py::ssize_t dims) {
    if (type == 0) {
        double difference[3] = {0.0, 0.0, 0.0};
        for (py::ssize_t index = 0; index < dims; ++index) {
            difference[index] = point[index] - params[index];
        }
        return norm(difference, dims) - params[3];
    }
    if (type == 1) {
        return point_box_clearance(point, params, params + 3, dims);
    }
    return point_segment_distance(point, params, params + 3, dims) - params[6];
}

static double segment_clearance(
    const double* start,
    const double* end,
    const double* params,
    std::int8_t type,
    py::ssize_t dims,
    py::ssize_t box_samples) {
    if (type == 0) {
        return point_segment_distance(params, start, end, dims) - params[3];
    }
    if (type == 2) {
        return segment_distance(start, end, params, params + 3, dims) - params[6];
    }
    double minimum = std::numeric_limits<double>::infinity();
    for (py::ssize_t sample = 0; sample < box_samples; ++sample) {
        const double alpha = static_cast<double>(sample) / static_cast<double>(box_samples - 1);
        double point[3] = {0.0, 0.0, 0.0};
        for (py::ssize_t index = 0; index < dims; ++index) {
            point[index] = start[index] + alpha * (end[index] - start[index]);
        }
        minimum = std::min(minimum, point_clearance(point, params, type, dims));
    }
    return minimum;
}

static void validate_inputs(
    const py::buffer_info& points,
    const py::buffer_info& types,
    const py::buffer_info& dims,
    const py::buffer_info& params,
    py::ssize_t box_samples) {
    if (points.ndim != 3 || points.shape[2] < 1 || points.shape[2] > 3) {
        throw std::invalid_argument("points must have shape (n_samples, n_points, 1..3)");
    }
    if (types.ndim != 1 || dims.ndim != 1 || types.shape[0] != dims.shape[0]) {
        throw std::invalid_argument("obstacle types and dimensions must be matching 1D arrays");
    }
    if (params.ndim != 2 || params.shape[0] != types.shape[0] || params.shape[1] < 7) {
        throw std::invalid_argument("obstacle parameters must have shape (n_obstacles, 7)");
    }
    if (box_samples < 2) {
        throw std::invalid_argument("box_samples must be at least 2");
    }
}

static py::array_t<double> chain_clearance_batch(
    Points points_, Types types_, Types dims_, Parameters params_, double link_radius, py::ssize_t box_samples) {
    const auto points = points_.request();
    const auto types = types_.request();
    const auto dims = dims_.request();
    const auto params = params_.request();
    validate_inputs(points, types, dims, params, box_samples);
    if (!std::isfinite(link_radius) || link_radius < 0.0) {
        throw std::invalid_argument("link_radius must be a non-negative finite value");
    }

    const py::ssize_t samples = points.shape[0];
    const py::ssize_t n_points = points.shape[1];
    const py::ssize_t point_dims = points.shape[2];
    const py::ssize_t n_obstacles = types.shape[0];
    const auto* point_data = static_cast<const double*>(points.ptr);
    const auto* type_data = static_cast<const std::int8_t*>(types.ptr);
    const auto* dim_data = static_cast<const std::int8_t*>(dims.ptr);
    const auto* parameter_data = static_cast<const double*>(params.ptr);
    for (py::ssize_t index = 0; index < n_obstacles; ++index) {
        if (type_data[index] < 0 || type_data[index] > 2) {
            throw std::invalid_argument("obstacle type must be 0, 1, or 2");
        }
        if (dim_data[index] < 1 || dim_data[index] > point_dims) {
            throw std::invalid_argument("obstacle dimensions must fit inside points");
        }
    }

    py::array_t<double> output(py::array::ShapeContainer(std::vector<py::ssize_t>{samples}));
    auto* output_data = static_cast<double*>(output.request().ptr);
    {
        py::gil_scoped_release release;
        for (py::ssize_t sample = 0; sample < samples; ++sample) {
            double minimum = std::numeric_limits<double>::infinity();
            for (py::ssize_t obstacle = 0; obstacle < n_obstacles; ++obstacle) {
                const py::ssize_t obstacle_dims = dim_data[obstacle];
                const double* obstacle_params = parameter_data + obstacle * params.shape[1];
                const std::int8_t obstacle_type = type_data[obstacle];
                for (py::ssize_t point_index = 0; point_index < n_points; ++point_index) {
                    const double* point = point_data + (sample * n_points + point_index) * point_dims;
                    minimum = std::min(minimum, point_clearance(point, obstacle_params, obstacle_type, obstacle_dims));
                }
                for (py::ssize_t segment_index = 0; segment_index + 1 < n_points; ++segment_index) {
                    const double* start = point_data + (sample * n_points + segment_index) * point_dims;
                    const double* end = point_data + (sample * n_points + segment_index + 1) * point_dims;
                    minimum = std::min(
                        minimum,
                        segment_clearance(start, end, obstacle_params, obstacle_type, obstacle_dims, box_samples));
                }
            }
            output_data[sample] = minimum == std::numeric_limits<double>::infinity()
                ? minimum
                : minimum - link_radius;
        }
    }
    return output;
}

PYBIND11_MODULE(_collision_cpp, m) {
    m.doc() = "C++ accelerated batched robot collision clearance";
    m.def(
        "chain_clearance_batch", &chain_clearance_batch,
        py::arg("points"), py::arg("obstacle_types"), py::arg("obstacle_dims"),
        py::arg("obstacle_params"), py::arg("link_radius") = 0.0,
        py::arg("box_samples") = 25);
}
