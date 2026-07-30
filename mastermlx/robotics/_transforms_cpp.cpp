#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

using MatrixBatch = py::array_t<double, py::array::c_style | py::array::forcecast>;
using Matrix = py::array_t<double, py::array::c_style | py::array::forcecast>;
using Vector = py::array_t<double, py::array::c_style | py::array::forcecast>;

static void require_finite(const double* values, py::ssize_t size, const char* name) {
    for (py::ssize_t index = 0; index < size; ++index) {
        if (!std::isfinite(values[index])) {
            throw std::invalid_argument(std::string(name) + " must contain only finite values");
        }
    }
}

static py::array_t<double> output_array(
    py::object requested, const std::vector<py::ssize_t>& shape) {
    if (requested.is_none()) {
        return py::array_t<double>(py::array::ShapeContainer(shape));
    }
    auto output = py::array_t<double, py::array::c_style>::ensure(requested);
    if (!output) {
        throw std::invalid_argument("output must be a contiguous float64 NumPy array");
    }
    const auto info = output.request();
    if (info.ndim != static_cast<py::ssize_t>(shape.size())) {
        throw std::invalid_argument("output has an unexpected number of dimensions");
    }
    for (std::size_t index = 0; index < shape.size(); ++index) {
        if (info.shape[index] != shape[index]) {
            throw std::invalid_argument("output has an unexpected shape");
        }
    }
    return output;
}

static void matmul4(const double* first, const double* second, double* result) {
    double product[16];
    for (int row = 0; row < 4; ++row) {
        for (int column = 0; column < 4; ++column) {
            product[row * 4 + column] =
                first[row * 4] * second[column]
                + first[row * 4 + 1] * second[4 + column]
                + first[row * 4 + 2] * second[8 + column]
                + first[row * 4 + 3] * second[12 + column];
        }
    }
    std::copy(product, product + 16, result);
}

static void matrix_to_quaternion(const double* matrix, double* quaternion) {
    const double trace = matrix[0] + matrix[5] + matrix[10];
    double s;
    if (trace > 0.0) {
        s = std::sqrt(trace + 1.0) * 2.0;
        quaternion[0] = 0.25 * s;
        quaternion[1] = (matrix[9] - matrix[6]) / s;
        quaternion[2] = (matrix[2] - matrix[8]) / s;
        quaternion[3] = (matrix[4] - matrix[1]) / s;
    } else if (matrix[0] > matrix[5] && matrix[0] > matrix[10]) {
        s = std::sqrt(1.0 + matrix[0] - matrix[5] - matrix[10]) * 2.0;
        quaternion[0] = (matrix[9] - matrix[6]) / s;
        quaternion[1] = 0.25 * s;
        quaternion[2] = (matrix[1] + matrix[4]) / s;
        quaternion[3] = (matrix[2] + matrix[8]) / s;
    } else if (matrix[5] > matrix[10]) {
        s = std::sqrt(1.0 + matrix[5] - matrix[0] - matrix[10]) * 2.0;
        quaternion[0] = (matrix[2] - matrix[8]) / s;
        quaternion[1] = (matrix[1] + matrix[4]) / s;
        quaternion[2] = 0.25 * s;
        quaternion[3] = (matrix[6] + matrix[9]) / s;
    } else {
        s = std::sqrt(1.0 + matrix[10] - matrix[0] - matrix[5]) * 2.0;
        quaternion[0] = (matrix[4] - matrix[1]) / s;
        quaternion[1] = (matrix[2] + matrix[8]) / s;
        quaternion[2] = (matrix[6] + matrix[9]) / s;
        quaternion[3] = 0.25 * s;
    }
    const double norm = std::sqrt(
        quaternion[0] * quaternion[0] + quaternion[1] * quaternion[1]
        + quaternion[2] * quaternion[2] + quaternion[3] * quaternion[3]);
    if (norm == 0.0 || !std::isfinite(norm)) {
        throw std::invalid_argument("rotation matrix produced an invalid quaternion");
    }
    for (int index = 0; index < 4; ++index) {
        quaternion[index] /= norm;
    }
}

static void quaternion_to_matrix(const double* quaternion, double* matrix) {
    const double w = quaternion[0];
    const double x = quaternion[1];
    const double y = quaternion[2];
    const double z = quaternion[3];
    matrix[0] = 1.0 - 2.0 * (y * y + z * z);
    matrix[1] = 2.0 * (x * y - z * w);
    matrix[2] = 2.0 * (x * z + y * w);
    matrix[3] = 0.0;
    matrix[4] = 2.0 * (x * y + z * w);
    matrix[5] = 1.0 - 2.0 * (x * x + z * z);
    matrix[6] = 2.0 * (y * z - x * w);
    matrix[7] = 0.0;
    matrix[8] = 2.0 * (x * z - y * w);
    matrix[9] = 2.0 * (y * z + x * w);
    matrix[10] = 1.0 - 2.0 * (x * x + y * y);
    matrix[11] = 0.0;
    matrix[12] = 0.0;
    matrix[13] = 0.0;
    matrix[14] = 0.0;
    matrix[15] = 1.0;
}

static py::array_t<double> compose_transform_batch(
    MatrixBatch transforms, py::object requested_output) {
    const auto input = transforms.request();
    if (input.ndim != 4 || input.shape[1] < 1 || input.shape[2] != 4 || input.shape[3] != 4) {
        throw std::invalid_argument("transforms must have shape (n_samples, n_transforms, 4, 4)");
    }
    require_finite(static_cast<const double*>(input.ptr), input.size, "transforms");
    const py::ssize_t samples = input.shape[0];
    const py::ssize_t count = input.shape[1];
    auto output = output_array(requested_output, {samples, 4, 4});
    auto* output_data = static_cast<double*>(output.request().ptr);
    const auto* input_data = static_cast<const double*>(input.ptr);

    py::gil_scoped_release release;
    for (py::ssize_t sample = 0; sample < samples; ++sample) {
        double accumulated[16];
        std::copy(input_data + (sample * count) * 16, input_data + (sample * count + 1) * 16, accumulated);
        for (py::ssize_t transform = 1; transform < count; ++transform) {
            matmul4(
                accumulated,
                input_data + (sample * count + transform) * 16,
                accumulated);
        }
        std::copy(accumulated, accumulated + 16, output_data + sample * 16);
    }
    return output;
}

static py::array_t<double> interpolate_pose_batch(
    Matrix start, Matrix end, Vector alphas, py::object requested_output) {
    const auto start_info = start.request();
    const auto end_info = end.request();
    const auto alpha_info = alphas.request();
    if (start_info.ndim != 2 || start_info.shape[0] != 4 || start_info.shape[1] != 4
        || end_info.ndim != 2 || end_info.shape[0] != 4 || end_info.shape[1] != 4
        || alpha_info.ndim != 1 || alpha_info.shape[0] < 1) {
        throw std::invalid_argument("start and end must be 4x4 and alphas must be a non-empty vector");
    }
    require_finite(static_cast<const double*>(start_info.ptr), 16, "start");
    require_finite(static_cast<const double*>(end_info.ptr), 16, "end");
    require_finite(static_cast<const double*>(alpha_info.ptr), alpha_info.shape[0], "alphas");
    auto output = output_array(requested_output, {alpha_info.shape[0], 4, 4});
    auto* output_data = static_cast<double*>(output.request().ptr);
    const auto* start_data = static_cast<const double*>(start_info.ptr);
    const auto* end_data = static_cast<const double*>(end_info.ptr);
    const auto* alpha_data = static_cast<const double*>(alpha_info.ptr);
    double first[4];
    double second[4];
    matrix_to_quaternion(start_data, first);
    matrix_to_quaternion(end_data, second);
    double dot = first[0] * second[0] + first[1] * second[1] + first[2] * second[2] + first[3] * second[3];
    if (dot < 0.0) {
        dot = -dot;
        for (double& value : second) {
            value = -value;
        }
    }
    dot = std::max(-1.0, std::min(1.0, dot));

    py::gil_scoped_release release;
    for (py::ssize_t sample = 0; sample < alpha_info.shape[0]; ++sample) {
        const double alpha = alpha_data[sample];
        double quaternion[4];
        if (dot > 0.9995) {
            double norm = 0.0;
            for (int index = 0; index < 4; ++index) {
                quaternion[index] = first[index] + alpha * (second[index] - first[index]);
                norm += quaternion[index] * quaternion[index];
            }
            norm = std::sqrt(norm);
            for (double& value : quaternion) {
                value /= norm;
            }
        } else {
            const double angle = std::acos(dot);
            const double denominator = std::sin(angle);
            const double first_weight = std::sin((1.0 - alpha) * angle) / denominator;
            const double second_weight = std::sin(alpha * angle) / denominator;
            for (int index = 0; index < 4; ++index) {
                quaternion[index] = first_weight * first[index] + second_weight * second[index];
            }
        }
        double* pose = output_data + sample * 16;
        quaternion_to_matrix(quaternion, pose);
        for (int index = 0; index < 3; ++index) {
            pose[index * 4 + 3] = start_data[index * 4 + 3]
                + alpha * (end_data[index * 4 + 3] - start_data[index * 4 + 3]);
        }
    }
    return output;
}

PYBIND11_MODULE(_transforms_cpp, m) {
    m.doc() = "C++ accelerated batched robotics transform helpers";
    m.def(
        "compose_transform_batch", &compose_transform_batch,
        py::arg("transforms"), py::arg("output") = py::none());
    m.def(
        "interpolate_pose_batch", &interpolate_pose_batch,
        py::arg("start"), py::arg("end"), py::arg("alphas"),
        py::arg("output") = py::none());
}
