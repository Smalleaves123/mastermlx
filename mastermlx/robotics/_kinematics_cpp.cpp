#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace py = pybind11;

using Vector = py::array_t<double, py::array::c_style | py::array::forcecast>;
using JointTypes = py::array_t<std::int8_t, py::array::c_style | py::array::forcecast>;
using Batch = py::array_t<double, py::array::c_style | py::array::forcecast>;
using Transform = std::array<double, 16>;

static void require_vector(const py::buffer_info& info, const char* name) {
    if (info.ndim != 1) {
        throw std::invalid_argument(std::string(name) + " must be a 1D array");
    }
}

static Transform read_transform(py::handle value, const char* name) {
    Transform result{};
    if (value.is_none()) {
        result = {1.0, 0.0, 0.0, 0.0,
                  0.0, 1.0, 0.0, 0.0,
                  0.0, 0.0, 1.0, 0.0,
                  0.0, 0.0, 0.0, 1.0};
        return result;
    }
    Vector array = Vector::ensure(value);
    if (!array) {
        throw std::invalid_argument(std::string(name) + " must be array-like");
    }
    const auto info = array.request();
    if (info.ndim != 2 || info.shape[0] != 4 || info.shape[1] != 4) {
        throw std::invalid_argument(std::string(name) + " must have shape (4, 4)");
    }
    const auto* data = static_cast<const double*>(info.ptr);
    std::copy(data, data + 16, result.begin());
    return result;
}

static void validate_inputs(
    const py::buffer_info& a,
    const py::buffer_info& alpha,
    const py::buffer_info& d,
    const py::buffer_info& theta,
    const py::buffer_info& joint_type,
    const py::buffer_info& offset,
    const py::buffer_info& q) {
    require_vector(a, "a");
    require_vector(alpha, "alpha");
    require_vector(d, "d");
    require_vector(theta, "theta");
    require_vector(joint_type, "joint_type");
    require_vector(offset, "offset");
    if (alpha.shape[0] != a.shape[0] || d.shape[0] != a.shape[0]
        || theta.shape[0] != a.shape[0] || joint_type.shape[0] != a.shape[0]
        || offset.shape[0] != a.shape[0]) {
        throw std::invalid_argument("DH parameter arrays must have the same length");
    }
    if (q.ndim != 2 || q.shape[1] != a.shape[0]) {
        throw std::invalid_argument("q must have shape (n_samples, n_joints)");
    }
}

static inline void matmul4(const double* left, const double* right, double* out) {
    double value[16];
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            value[row * 4 + col] =
                left[row * 4] * right[col]
                + left[row * 4 + 1] * right[4 + col]
                + left[row * 4 + 2] * right[8 + col]
                + left[row * 4 + 3] * right[12 + col];
        }
    }
    std::copy(value, value + 16, out);
}

static void chain_transform(
    const double* a,
    const double* alpha,
    const double* d,
    const double* theta,
    const std::int8_t* joint_type,
    const double* offset,
    const double* q,
    py::ssize_t n,
    const Transform& base,
    const Transform* tool,
    double* output,
    double* origins,
    double* axes) {
    double transform[16];
    double next[16];
    std::copy(base.begin(), base.end(), transform);

    for (py::ssize_t i = 0; i < n; ++i) {
        if (origins != nullptr) {
            origins[3 * i] = transform[3];
            origins[3 * i + 1] = transform[7];
            origins[3 * i + 2] = transform[11];
            axes[3 * i] = transform[2];
            axes[3 * i + 1] = transform[6];
            axes[3 * i + 2] = transform[10];
        }

        double joint_d = d[i];
        double joint_theta = theta[i];
        if (joint_type[i] != 0) {
            joint_theta += q[i] + offset[i];
        } else {
            joint_d += q[i] + offset[i];
        }

        const double ct = std::cos(joint_theta);
        const double st = std::sin(joint_theta);
        const double ca = std::cos(alpha[i]);
        const double sa = std::sin(alpha[i]);
        const double link[16] = {
            ct, -st * ca, st * sa, a[i] * ct,
            st, ct * ca, -ct * sa, a[i] * st,
            0.0, sa, ca, joint_d,
            0.0, 0.0, 0.0, 1.0,
        };
        matmul4(transform, link, next);
        std::copy(next, next + 16, transform);
    }

    if (tool != nullptr) {
        matmul4(transform, tool->data(), next);
        std::copy(next, next + 16, transform);
    }
    std::copy(transform, transform + 16, output);
}

template <typename Fn>
static void parallel_rows(py::ssize_t rows, py::ssize_t work, Fn&& fn) {
    if (rows < 4 || work < 1 || static_cast<long double>(rows) * work < 4096.0L) {
        fn(0, rows);
        return;
    }
    const unsigned hardware = std::thread::hardware_concurrency();
    const unsigned available = hardware == 0 ? 1u : hardware;
    const unsigned workers = std::min<unsigned>(std::min<unsigned>(available, 8u), static_cast<unsigned>(rows));
    if (workers <= 1) {
        fn(0, rows);
        return;
    }
    const py::ssize_t chunk = (rows + workers - 1) / workers;
    std::vector<std::thread> threads;
    threads.reserve(workers);
    for (unsigned worker = 0; worker < workers; ++worker) {
        const py::ssize_t begin = worker * chunk;
        const py::ssize_t end = std::min(rows, begin + chunk);
        if (begin < end) {
            threads.emplace_back([&fn, begin, end]() { fn(begin, end); });
        }
    }
    for (auto& thread : threads) {
        thread.join();
    }
}

static py::array_t<double> forward_kinematics_batch_dh(
    Vector a_, Vector alpha_, Vector d_, Vector theta_, JointTypes joint_type_, Vector offset_,
    Batch q_, py::object base, py::object tool) {
    const auto a = a_.request();
    const auto alpha = alpha_.request();
    const auto d = d_.request();
    const auto theta = theta_.request();
    const auto joint_type = joint_type_.request();
    const auto offset = offset_.request();
    const auto q = q_.request();
    validate_inputs(a, alpha, d, theta, joint_type, offset, q);
    const Transform base_transform = read_transform(base, "base");
    const bool has_tool = !tool.is_none();
    const Transform tool_transform = read_transform(tool, "tool");
    const py::ssize_t samples = q.shape[0];
    const py::ssize_t joints = a.shape[0];
    py::array_t<double> output(
        py::array::ShapeContainer(std::vector<py::ssize_t>{samples, 4, 4}));

    const auto* a_data = static_cast<const double*>(a.ptr);
    const auto* alpha_data = static_cast<const double*>(alpha.ptr);
    const auto* d_data = static_cast<const double*>(d.ptr);
    const auto* theta_data = static_cast<const double*>(theta.ptr);
    const auto* type_data = static_cast<const std::int8_t*>(joint_type.ptr);
    const auto* offset_data = static_cast<const double*>(offset.ptr);
    const auto* q_data = static_cast<const double*>(q.ptr);
    auto* output_data = static_cast<double*>(output.request().ptr);
    {
        py::gil_scoped_release release;
        parallel_rows(samples, joints, [&](py::ssize_t begin, py::ssize_t end) {
            for (py::ssize_t sample = begin; sample < end; ++sample) {
                chain_transform(
                    a_data, alpha_data, d_data, theta_data, type_data, offset_data,
                    q_data + sample * joints, joints, base_transform,
                    has_tool ? &tool_transform : nullptr,
                    output_data + sample * 16, nullptr, nullptr);
            }
        });
    }
    return output;
}

static py::array_t<double> geometric_jacobian_batch_dh(
    Vector a_, Vector alpha_, Vector d_, Vector theta_, JointTypes joint_type_, Vector offset_,
    Batch q_, py::object base, py::object tool) {
    const auto a = a_.request();
    const auto alpha = alpha_.request();
    const auto d = d_.request();
    const auto theta = theta_.request();
    const auto joint_type = joint_type_.request();
    const auto offset = offset_.request();
    const auto q = q_.request();
    validate_inputs(a, alpha, d, theta, joint_type, offset, q);
    const Transform base_transform = read_transform(base, "base");
    const bool has_tool = !tool.is_none();
    const Transform tool_transform = read_transform(tool, "tool");
    const py::ssize_t samples = q.shape[0];
    const py::ssize_t joints = a.shape[0];
    py::array_t<double> output(
        py::array::ShapeContainer(std::vector<py::ssize_t>{samples, 6, joints}));

    const auto* a_data = static_cast<const double*>(a.ptr);
    const auto* alpha_data = static_cast<const double*>(alpha.ptr);
    const auto* d_data = static_cast<const double*>(d.ptr);
    const auto* theta_data = static_cast<const double*>(theta.ptr);
    const auto* type_data = static_cast<const std::int8_t*>(joint_type.ptr);
    const auto* offset_data = static_cast<const double*>(offset.ptr);
    const auto* q_data = static_cast<const double*>(q.ptr);
    auto* output_data = static_cast<double*>(output.request().ptr);
    {
        py::gil_scoped_release release;
        parallel_rows(samples, joints, [&](py::ssize_t begin, py::ssize_t end) {
            std::vector<double> origins(static_cast<std::size_t>(3 * joints));
            std::vector<double> axes(static_cast<std::size_t>(3 * joints));
            for (py::ssize_t sample = begin; sample < end; ++sample) {
                double transform[16];
                chain_transform(
                    a_data, alpha_data, d_data, theta_data, type_data, offset_data,
                    q_data + sample * joints, joints, base_transform,
                    has_tool ? &tool_transform : nullptr,
                    transform, origins.data(), axes.data());
                const double px = transform[3];
                const double py = transform[7];
                const double pz = transform[11];
                double* jacobian = output_data + sample * 6 * joints;
                for (py::ssize_t i = 0; i < joints; ++i) {
                    const double dx = px - origins[3 * i];
                    const double dy = py - origins[3 * i + 1];
                    const double dz = pz - origins[3 * i + 2];
                    const double ax = axes[3 * i];
                    const double ay = axes[3 * i + 1];
                    const double az = axes[3 * i + 2];
                    if (type_data[i] != 0) {
                        jacobian[i] = ay * dz - az * dy;
                        jacobian[joints + i] = az * dx - ax * dz;
                        jacobian[2 * joints + i] = ax * dy - ay * dx;
                        jacobian[3 * joints + i] = ax;
                        jacobian[4 * joints + i] = ay;
                        jacobian[5 * joints + i] = az;
                    } else {
                        jacobian[i] = ax;
                        jacobian[joints + i] = ay;
                        jacobian[2 * joints + i] = az;
                        jacobian[3 * joints + i] = 0.0;
                        jacobian[4 * joints + i] = 0.0;
                        jacobian[5 * joints + i] = 0.0;
                    }
                }
            }
        });
    }
    return output;
}

PYBIND11_MODULE(_kinematics_cpp, m) {
    m.doc() = "C++ accelerated batched DH kinematics";
    m.def(
        "forward_kinematics_batch_dh", &forward_kinematics_batch_dh,
        py::arg("a"), py::arg("alpha"), py::arg("d"), py::arg("theta"),
        py::arg("joint_type"), py::arg("offset"), py::arg("q"),
        py::arg("base") = py::none(), py::arg("tool") = py::none());
    m.def(
        "geometric_jacobian_batch_dh", &geometric_jacobian_batch_dh,
        py::arg("a"), py::arg("alpha"), py::arg("d"), py::arg("theta"),
        py::arg("joint_type"), py::arg("offset"), py::arg("q"),
        py::arg("base") = py::none(), py::arg("tool") = py::none());
}
