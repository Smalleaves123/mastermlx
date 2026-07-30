#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

using Points = py::array_t<double, py::array::c_style | py::array::forcecast>;
using Parameters = py::array_t<double, py::array::c_style | py::array::forcecast>;
using Types = py::array_t<std::int8_t, py::array::c_style | py::array::forcecast>;

template <typename T>
static py::array_t<T> details_output(
    py::object requested,
    const std::vector<py::ssize_t>& shape,
    const char* name) {
    if (requested.is_none()) {
        return py::array_t<T>(py::array::ShapeContainer(shape));
    }
    auto output = py::array_t<T, py::array::c_style>::ensure(requested);
    if (!output) {
        throw std::invalid_argument(std::string(name) + " must be a contiguous array with the expected dtype");
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

static void obstacle_bounds(
    const double* params, std::int8_t type, py::ssize_t dims, double* lower, double* upper) {
    if (type == 1) {
        for (py::ssize_t index = 0; index < dims; ++index) {
            lower[index] = params[index];
            upper[index] = params[3 + index];
        }
        return;
    }
    if (type == 0) {
        const double radius = params[3];
        for (py::ssize_t index = 0; index < dims; ++index) {
            lower[index] = params[index] - radius;
            upper[index] = params[index] + radius;
        }
        return;
    }
    const double radius = params[6];
    for (py::ssize_t index = 0; index < dims; ++index) {
        const double first = params[index];
        const double second = params[3 + index];
        lower[index] = std::min(first, second) - radius;
        upper[index] = std::max(first, second) + radius;
    }
}

static double aabb_distance(
    const double* first_lower,
    const double* first_upper,
    const double* second_lower,
    const double* second_upper,
    py::ssize_t dims) {
    double squared = 0.0;
    for (py::ssize_t index = 0; index < dims; ++index) {
        double gap = 0.0;
        if (first_upper[index] < second_lower[index]) {
            gap = second_lower[index] - first_upper[index];
        } else if (second_upper[index] < first_lower[index]) {
            gap = first_lower[index] - second_upper[index];
        }
        squared += gap * gap;
    }
    return std::sqrt(squared);
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

static py::array_t<bool> chain_collision_free_batch(
    Points points_, Types types_, Types dims_, Parameters params_, double clearance,
    double link_radius, py::ssize_t box_samples) {
    const auto points = points_.request();
    const auto types = types_.request();
    const auto dims = dims_.request();
    const auto params = params_.request();
    validate_inputs(points, types, dims, params, box_samples);
    if (!std::isfinite(clearance) || !std::isfinite(link_radius) || clearance < 0.0 || link_radius < 0.0) {
        throw std::invalid_argument("clearance and link_radius must be non-negative finite values");
    }

    const py::ssize_t samples = points.shape[0];
    const py::ssize_t n_points = points.shape[1];
    const py::ssize_t point_dims = points.shape[2];
    const py::ssize_t n_obstacles = types.shape[0];
    const auto* point_data = static_cast<const double*>(points.ptr);
    const auto* type_data = static_cast<const std::int8_t*>(types.ptr);
    const auto* dim_data = static_cast<const std::int8_t*>(dims.ptr);
    const auto* parameter_data = static_cast<const double*>(params.ptr);
    py::array_t<bool> output(py::array::ShapeContainer(std::vector<py::ssize_t>{samples}));
    auto* output_data = static_cast<bool*>(output.request().ptr);

    {
        py::gil_scoped_release release;
        for (py::ssize_t sample = 0; sample < samples; ++sample) {
            bool free = true;
            for (py::ssize_t obstacle = 0; obstacle < n_obstacles && free; ++obstacle) {
                const py::ssize_t obstacle_dims = dim_data[obstacle];
                const std::int8_t obstacle_type = type_data[obstacle];
                const double* obstacle_params = parameter_data + obstacle * params.shape[1];
                double obstacle_lower[3] = {0.0, 0.0, 0.0};
                double obstacle_upper[3] = {0.0, 0.0, 0.0};
                obstacle_bounds(obstacle_params, obstacle_type, obstacle_dims, obstacle_lower, obstacle_upper);
                for (py::ssize_t point_index = 0; point_index < n_points && free; ++point_index) {
                    const double* point = point_data + (sample * n_points + point_index) * point_dims;
                    double point_lower[3] = {0.0, 0.0, 0.0};
                    double point_upper[3] = {0.0, 0.0, 0.0};
                    for (py::ssize_t index = 0; index < obstacle_dims; ++index) {
                        point_lower[index] = point[index];
                        point_upper[index] = point[index];
                    }
                    const double lower_bound = aabb_distance(
                        point_lower, point_upper, obstacle_lower, obstacle_upper, obstacle_dims) - link_radius;
                    if (lower_bound <= clearance
                        && point_clearance(point, obstacle_params, obstacle_type, obstacle_dims) - link_radius < clearance) {
                        free = false;
                    }
                }
                for (py::ssize_t segment = 0; segment + 1 < n_points && free; ++segment) {
                    const double* start = point_data + (sample * n_points + segment) * point_dims;
                    const double* end = point_data + (sample * n_points + segment + 1) * point_dims;
                    double segment_lower[3] = {0.0, 0.0, 0.0};
                    double segment_upper[3] = {0.0, 0.0, 0.0};
                    for (py::ssize_t index = 0; index < obstacle_dims; ++index) {
                        segment_lower[index] = std::min(start[index], end[index]);
                        segment_upper[index] = std::max(start[index], end[index]);
                    }
                    const double lower_bound = aabb_distance(
                        segment_lower, segment_upper, obstacle_lower, obstacle_upper, obstacle_dims) - link_radius;
                    if (lower_bound <= clearance
                        && segment_clearance(start, end, obstacle_params, obstacle_type, obstacle_dims, box_samples)
                            - link_radius < clearance) {
                        free = false;
                    }
                }
            }
            output_data[sample] = free;
        }
    }
    return output;
}

static py::tuple chain_collision_summary_batch(
    Points points_, Types types_, Types dims_, Parameters params_, double link_radius,
    py::ssize_t box_samples) {
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

    py::array_t<double> clearances(py::array::ShapeContainer(std::vector<py::ssize_t>{samples}));
    py::array_t<bool> collisions(py::array::ShapeContainer(std::vector<py::ssize_t>{samples}));
    py::array_t<std::int8_t> kinds(py::array::ShapeContainer(std::vector<py::ssize_t>{samples}));
    py::array_t<std::int64_t> indices(py::array::ShapeContainer(std::vector<py::ssize_t>{samples}));
    py::array_t<std::int64_t> obstacle_indices(
        py::array::ShapeContainer(std::vector<py::ssize_t>{samples}));
    auto* clearance_data = static_cast<double*>(clearances.request().ptr);
    auto* collision_data = static_cast<bool*>(collisions.request().ptr);
    auto* kind_data = static_cast<std::int8_t*>(kinds.request().ptr);
    auto* index_data = static_cast<std::int64_t*>(indices.request().ptr);
    auto* obstacle_index_data = static_cast<std::int64_t*>(obstacle_indices.request().ptr);

    {
        py::gil_scoped_release release;
        for (py::ssize_t sample = 0; sample < samples; ++sample) {
            double minimum = std::numeric_limits<double>::infinity();
            std::int8_t closest_kind = 0;
            std::int64_t closest_index = -1;
            std::int64_t closest_obstacle = -1;
            for (py::ssize_t obstacle = 0; obstacle < n_obstacles; ++obstacle) {
                const py::ssize_t obstacle_dims = dim_data[obstacle];
                const double* obstacle_params = parameter_data + obstacle * params.shape[1];
                const std::int8_t obstacle_type = type_data[obstacle];
                for (py::ssize_t point_index = 0; point_index < n_points; ++point_index) {
                    const double* point = point_data + (sample * n_points + point_index) * point_dims;
                    const double clearance = point_clearance(
                        point, obstacle_params, obstacle_type, obstacle_dims) - link_radius;
                    if (clearance < minimum) {
                        minimum = clearance;
                        closest_kind = 1;
                        closest_index = point_index;
                        closest_obstacle = obstacle;
                    }
                }
                for (py::ssize_t segment = 0; segment + 1 < n_points; ++segment) {
                    const double* start = point_data + (sample * n_points + segment) * point_dims;
                    const double* end = point_data + (sample * n_points + segment + 1) * point_dims;
                    const double clearance = segment_clearance(
                        start, end, obstacle_params, obstacle_type, obstacle_dims, box_samples)
                        - link_radius;
                    if (clearance < minimum) {
                        minimum = clearance;
                        closest_kind = 2;
                        closest_index = segment;
                        closest_obstacle = obstacle;
                    }
                }
            }
            clearance_data[sample] = minimum;
            collision_data[sample] = minimum <= 0.0;
            kind_data[sample] = closest_kind;
            index_data[sample] = closest_index;
            obstacle_index_data[sample] = closest_obstacle;
        }
    }
    return py::make_tuple(clearances, collisions, kinds, indices, obstacle_indices);
}

static py::tuple chain_collision_details_batch(
    Points points_, Types types_, Types dims_, Parameters params_, double link_radius,
    py::ssize_t box_samples, py::ssize_t max_hits,
    py::object requested_clearances, py::object requested_collisions,
    py::object requested_closest_kinds, py::object requested_closest_indices,
    py::object requested_closest_obstacles, py::object requested_hit_counts,
    py::object requested_hit_truncated, py::object requested_hit_kinds,
    py::object requested_hit_indices, py::object requested_hit_obstacles,
    py::object requested_hit_clearances) {
    const auto points = points_.request();
    const auto types = types_.request();
    const auto dims = dims_.request();
    const auto params = params_.request();
    validate_inputs(points, types, dims, params, box_samples);
    if (!std::isfinite(link_radius) || link_radius < 0.0) {
        throw std::invalid_argument("link_radius must be a non-negative finite value");
    }
    if (max_hits < 0) {
        throw std::invalid_argument("max_hits must be non-negative");
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

    const std::vector<py::ssize_t> sample_shape{samples};
    const std::vector<py::ssize_t> hit_shape{samples, max_hits};
    auto clearances = details_output<double>(requested_clearances, sample_shape, "minimum_clearance");
    auto collisions = details_output<bool>(requested_collisions, sample_shape, "collision");
    auto closest_kinds = details_output<std::int8_t>(
        requested_closest_kinds, sample_shape, "closest_kind");
    auto closest_indices = details_output<std::int64_t>(
        requested_closest_indices, sample_shape, "closest_index");
    auto closest_obstacles = details_output<std::int64_t>(
        requested_closest_obstacles, sample_shape, "closest_obstacle_index");
    auto hit_counts = details_output<std::int64_t>(requested_hit_counts, sample_shape, "hit_count");
    auto hit_truncated = details_output<bool>(
        requested_hit_truncated, sample_shape, "hit_truncated");
    auto hit_kinds = details_output<std::int8_t>(requested_hit_kinds, hit_shape, "hit_kind");
    auto hit_indices = details_output<std::int64_t>(requested_hit_indices, hit_shape, "hit_index");
    auto hit_obstacles = details_output<std::int64_t>(
        requested_hit_obstacles, hit_shape, "hit_obstacle_index");
    auto hit_clearances = details_output<double>(
        requested_hit_clearances, hit_shape, "hit_clearance");
    auto* clearance_data = static_cast<double*>(clearances.request().ptr);
    auto* collision_data = static_cast<bool*>(collisions.request().ptr);
    auto* closest_kind_data = static_cast<std::int8_t*>(closest_kinds.request().ptr);
    auto* closest_index_data = static_cast<std::int64_t*>(closest_indices.request().ptr);
    auto* closest_obstacle_data = static_cast<std::int64_t*>(closest_obstacles.request().ptr);
    auto* hit_count_data = static_cast<std::int64_t*>(hit_counts.request().ptr);
    auto* hit_truncated_data = static_cast<bool*>(hit_truncated.request().ptr);
    auto* hit_kind_data = static_cast<std::int8_t*>(hit_kinds.request().ptr);
    auto* hit_index_data = static_cast<std::int64_t*>(hit_indices.request().ptr);
    auto* hit_obstacle_data = static_cast<std::int64_t*>(hit_obstacles.request().ptr);
    auto* hit_clearance_data = static_cast<double*>(hit_clearances.request().ptr);

    {
        py::gil_scoped_release release;
        for (py::ssize_t sample = 0; sample < samples; ++sample) {
            double minimum = std::numeric_limits<double>::infinity();
            std::int8_t closest_kind = 0;
            std::int64_t closest_index = -1;
            std::int64_t closest_obstacle = -1;
            std::int64_t hit_count = 0;
            bool truncated = false;
            for (py::ssize_t slot = 0; slot < max_hits; ++slot) {
                const py::ssize_t offset = sample * max_hits + slot;
                hit_kind_data[offset] = 0;
                hit_index_data[offset] = -1;
                hit_obstacle_data[offset] = -1;
                hit_clearance_data[offset] = std::numeric_limits<double>::infinity();
            }
            for (py::ssize_t obstacle = 0; obstacle < n_obstacles; ++obstacle) {
                const py::ssize_t obstacle_dims = dim_data[obstacle];
                const double* obstacle_params = parameter_data + obstacle * params.shape[1];
                const std::int8_t obstacle_type = type_data[obstacle];
                for (py::ssize_t point_index = 0; point_index < n_points; ++point_index) {
                    const double* point = point_data + (sample * n_points + point_index) * point_dims;
                    const double clearance = point_clearance(
                        point, obstacle_params, obstacle_type, obstacle_dims) - link_radius;
                    if (clearance < minimum) {
                        minimum = clearance;
                        closest_kind = 1;
                        closest_index = point_index;
                        closest_obstacle = obstacle;
                    }
                    if (clearance <= 0.0) {
                        if (hit_count < max_hits) {
                            const py::ssize_t offset = sample * max_hits + hit_count;
                            hit_kind_data[offset] = 1;
                            hit_index_data[offset] = point_index;
                            hit_obstacle_data[offset] = obstacle;
                            hit_clearance_data[offset] = clearance;
                        } else {
                            truncated = true;
                        }
                        ++hit_count;
                    }
                }
                for (py::ssize_t segment = 0; segment + 1 < n_points; ++segment) {
                    const double* start = point_data + (sample * n_points + segment) * point_dims;
                    const double* end = point_data + (sample * n_points + segment + 1) * point_dims;
                    const double clearance = segment_clearance(
                        start, end, obstacle_params, obstacle_type, obstacle_dims, box_samples)
                        - link_radius;
                    if (clearance < minimum) {
                        minimum = clearance;
                        closest_kind = 2;
                        closest_index = segment;
                        closest_obstacle = obstacle;
                    }
                    if (clearance <= 0.0) {
                        if (hit_count < max_hits) {
                            const py::ssize_t offset = sample * max_hits + hit_count;
                            hit_kind_data[offset] = 2;
                            hit_index_data[offset] = segment;
                            hit_obstacle_data[offset] = obstacle;
                            hit_clearance_data[offset] = clearance;
                        } else {
                            truncated = true;
                        }
                        ++hit_count;
                    }
                }
            }
            clearance_data[sample] = minimum;
            collision_data[sample] = hit_count > 0;
            closest_kind_data[sample] = closest_kind;
            closest_index_data[sample] = closest_index;
            closest_obstacle_data[sample] = closest_obstacle;
            hit_count_data[sample] = hit_count;
            hit_truncated_data[sample] = truncated;
        }
    }
    return py::make_tuple(
        clearances, collisions, closest_kinds, closest_indices, closest_obstacles,
        hit_counts, hit_truncated, hit_kinds, hit_indices, hit_obstacles, hit_clearances);
}

PYBIND11_MODULE(_collision_cpp, m) {
    m.doc() = "C++ accelerated batched robot collision clearance";
    m.def(
        "chain_clearance_batch", &chain_clearance_batch,
        py::arg("points"), py::arg("obstacle_types"), py::arg("obstacle_dims"),
        py::arg("obstacle_params"), py::arg("link_radius") = 0.0,
        py::arg("box_samples") = 25);
    m.def(
        "chain_collision_free_batch", &chain_collision_free_batch,
        py::arg("points"), py::arg("obstacle_types"), py::arg("obstacle_dims"),
        py::arg("obstacle_params"), py::arg("clearance") = 0.0,
        py::arg("link_radius") = 0.0, py::arg("box_samples") = 25);
    m.def(
        "chain_collision_summary_batch", &chain_collision_summary_batch,
        py::arg("points"), py::arg("obstacle_types"), py::arg("obstacle_dims"),
        py::arg("obstacle_params"), py::arg("link_radius") = 0.0,
        py::arg("box_samples") = 25);
    m.def(
        "chain_collision_details_batch", &chain_collision_details_batch,
        py::arg("points"), py::arg("obstacle_types"), py::arg("obstacle_dims"),
        py::arg("obstacle_params"), py::arg("link_radius") = 0.0,
        py::arg("box_samples") = 25, py::arg("max_hits") = 0,
        py::arg("minimum_clearance") = py::none(), py::arg("collision") = py::none(),
        py::arg("closest_kind") = py::none(), py::arg("closest_index") = py::none(),
        py::arg("closest_obstacle_index") = py::none(), py::arg("hit_count") = py::none(),
        py::arg("hit_truncated") = py::none(), py::arg("hit_kind") = py::none(),
        py::arg("hit_index") = py::none(), py::arg("hit_obstacle_index") = py::none(),
        py::arg("hit_clearance") = py::none());
}
