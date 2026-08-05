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
using Batch = py::array_t<double, py::array::c_style | py::array::forcecast>;
using JointTypes = py::array_t<std::int8_t, py::array::c_style | py::array::forcecast>;
using Transform = std::array<double, 16>;

static void require_finite(const double* values, py::ssize_t size, const char* name) {
    for (py::ssize_t index = 0; index < size; ++index) {
        if (!std::isfinite(values[index])) {
            throw std::invalid_argument(std::string(name) + " must contain only finite values");
        }
    }
}

static Transform read_transform(py::handle value, const char* name) {
    Transform result = {1.0, 0.0, 0.0, 0.0,
                        0.0, 1.0, 0.0, 0.0,
                        0.0, 0.0, 1.0, 0.0,
                        0.0, 0.0, 0.0, 1.0};
    if (value.is_none()) {
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
    require_finite(data, 16, name);
    std::copy(data, data + 16, result.begin());
    return result;
}

static void matmul4(const double* left, const double* right, double* output) {
    double value[16];
    for (int row = 0; row < 4; ++row) {
        for (int column = 0; column < 4; ++column) {
            value[row * 4 + column] = left[row * 4] * right[column]
                + left[row * 4 + 1] * right[4 + column]
                + left[row * 4 + 2] * right[8 + column]
                + left[row * 4 + 3] * right[12 + column];
        }
    }
    std::copy(value, value + 16, output);
}

static void identity4(double* output) {
    std::fill(output, output + 16, 0.0);
    output[0] = 1.0;
    output[5] = 1.0;
    output[10] = 1.0;
    output[15] = 1.0;
}

static void origin_transform(const double* xyz, const double* rpy, double* output) {
    const double cr = std::cos(rpy[0]);
    const double sr = std::sin(rpy[0]);
    const double cp = std::cos(rpy[1]);
    const double sp = std::sin(rpy[1]);
    const double cy = std::cos(rpy[2]);
    const double sy = std::sin(rpy[2]);
    output[0] = cy * cp;
    output[1] = cy * sp * sr - sy * cr;
    output[2] = cy * sp * cr + sy * sr;
    output[3] = xyz[0];
    output[4] = sy * cp;
    output[5] = sy * sp * sr + cy * cr;
    output[6] = sy * sp * cr - cy * sr;
    output[7] = xyz[1];
    output[8] = -sp;
    output[9] = cp * sr;
    output[10] = cp * cr;
    output[11] = xyz[2];
    output[12] = 0.0;
    output[13] = 0.0;
    output[14] = 0.0;
    output[15] = 1.0;
}

static void axis_angle_transform(const double* axis, double angle, double* output) {
    const double norm = std::sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]);
    if (norm <= 1e-12 || !std::isfinite(norm)) {
        throw std::invalid_argument("URDF joint axis must be a non-zero finite vector");
    }
    const double x = axis[0] / norm;
    const double y = axis[1] / norm;
    const double z = axis[2] / norm;
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    const double one_minus_c = 1.0 - c;
    output[0] = c + x * x * one_minus_c;
    output[1] = x * y * one_minus_c - z * s;
    output[2] = x * z * one_minus_c + y * s;
    output[3] = 0.0;
    output[4] = y * x * one_minus_c + z * s;
    output[5] = c + y * y * one_minus_c;
    output[6] = y * z * one_minus_c - x * s;
    output[7] = 0.0;
    output[8] = z * x * one_minus_c - y * s;
    output[9] = z * y * one_minus_c + x * s;
    output[10] = c + z * z * one_minus_c;
    output[11] = 0.0;
    output[12] = 0.0;
    output[13] = 0.0;
    output[14] = 0.0;
    output[15] = 1.0;
}

static void translation_transform(const double* axis, double distance, double* output) {
    identity4(output);
    const double norm = std::sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]);
    if (norm <= 1e-12 || !std::isfinite(norm)) {
        throw std::invalid_argument("URDF joint axis must be a non-zero finite vector");
    }
    output[3] = axis[0] * distance / norm;
    output[7] = axis[1] * distance / norm;
    output[11] = axis[2] * distance / norm;
}

static void cross3(const double* first, const double* second, double* output) {
    output[0] = first[1] * second[2] - first[2] * second[1];
    output[1] = first[2] * second[0] - first[0] * second[2];
    output[2] = first[0] * second[1] - first[1] * second[0];
}

static void add3(const double* first, const double* second, double* output) {
    output[0] = first[0] + second[0];
    output[1] = first[1] + second[1];
    output[2] = first[2] + second[2];
}

static void scale3(const double* value, double scale, double* output) {
    output[0] = value[0] * scale;
    output[1] = value[1] * scale;
    output[2] = value[2] * scale;
}

static void validate_chain_inputs(
    const py::buffer_info& origin_xyz,
    const py::buffer_info& origin_rpy,
    const py::buffer_info& axis,
    const py::buffer_info& joint_type,
    const py::buffer_info& q,
    py::ssize_t* active_count) {
    if (origin_xyz.ndim != 2 || origin_xyz.shape[1] != 3
        || origin_rpy.ndim != 2 || origin_rpy.shape[1] != 3
        || axis.ndim != 2 || axis.shape[1] != 3
        || joint_type.ndim != 1) {
        throw std::invalid_argument("URDF chain arrays have invalid shapes");
    }
    const py::ssize_t joints = origin_xyz.shape[0];
    if (origin_rpy.shape[0] != joints || axis.shape[0] != joints
        || joint_type.shape[0] != joints) {
        throw std::invalid_argument("URDF chain arrays must have the same path length");
    }
    *active_count = 0;
    const auto* types = static_cast<const std::int8_t*>(joint_type.ptr);
    for (py::ssize_t index = 0; index < joints; ++index) {
        if (types[index] < 0 || types[index] > 2) {
            throw std::invalid_argument("URDF joint type codes must be 0, 1, or 2");
        }
        if (types[index] != 0) {
            ++(*active_count);
        }
    }
    if (q.ndim != 2 || q.shape[0] < 1 || q.shape[1] != *active_count) {
        throw std::invalid_argument("q must have shape (n_samples, n_active_joints)");
    }
    require_finite(static_cast<const double*>(origin_xyz.ptr), origin_xyz.size, "origin_xyz");
    require_finite(static_cast<const double*>(origin_rpy.ptr), origin_rpy.size, "origin_rpy");
    require_finite(static_cast<const double*>(axis.ptr), axis.size, "axis");
    require_finite(static_cast<const double*>(q.ptr), q.size, "q");
    const auto* axes = static_cast<const double*>(axis.ptr);
    for (py::ssize_t index = 0; index < joints; ++index) {
        if (types[index] == 0) {
            continue;
        }
        const double norm = std::sqrt(
            axes[3 * index] * axes[3 * index]
            + axes[3 * index + 1] * axes[3 * index + 1]
            + axes[3 * index + 2] * axes[3 * index + 2]);
        if (norm <= 1e-12 || !std::isfinite(norm)) {
            throw std::invalid_argument("URDF joint axis must be a non-zero finite vector");
        }
    }
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

static void chain_state(
    const double* origin_xyz,
    const double* origin_rpy,
    const double* axis,
    const std::int8_t* joint_type,
    const double* q,
    py::ssize_t joints,
    py::ssize_t active_count,
    const Transform& base,
    const Transform* tool,
    double* endpoint,
    double* frames,
    double* active_origins,
    double* active_axes) {
    double transform[16];
    std::copy(base.begin(), base.end(), transform);
    py::ssize_t active = 0;
    if (frames != nullptr) {
        std::copy(transform, transform + 16, frames);
    }
    for (py::ssize_t joint = 0; joint < joints; ++joint) {
        double origin[16];
        origin_transform(origin_xyz + 3 * joint, origin_rpy + 3 * joint, origin);
        double joint_frame[16];
        matmul4(transform, origin, joint_frame);
        double motion[16];
        identity4(motion);
        if (joint_type[joint] != 0) {
            const double norm = std::sqrt(
                axis[3 * joint] * axis[3 * joint]
                + axis[3 * joint + 1] * axis[3 * joint + 1]
                + axis[3 * joint + 2] * axis[3 * joint + 2]);
            const double local_axis[3] = {
                axis[3 * joint] / norm,
                axis[3 * joint + 1] / norm,
                axis[3 * joint + 2] / norm,
            };
            active_origins[3 * active] = joint_frame[3];
            active_origins[3 * active + 1] = joint_frame[7];
            active_origins[3 * active + 2] = joint_frame[11];
            active_axes[3 * active] = joint_frame[0] * local_axis[0]
                + joint_frame[1] * local_axis[1] + joint_frame[2] * local_axis[2];
            active_axes[3 * active + 1] = joint_frame[4] * local_axis[0]
                + joint_frame[5] * local_axis[1] + joint_frame[6] * local_axis[2];
            active_axes[3 * active + 2] = joint_frame[8] * local_axis[0]
                + joint_frame[9] * local_axis[1] + joint_frame[10] * local_axis[2];
            const double value = q[active];
            if (joint_type[joint] == 1) {
                axis_angle_transform(local_axis, value, motion);
            } else {
                translation_transform(local_axis, value, motion);
            }
            ++active;
        }
        matmul4(joint_frame, motion, transform);
        if (frames != nullptr) {
            std::copy(transform, transform + 16, frames + 16 * (joint + 1));
        }
    }
    if (tool != nullptr) {
        matmul4(transform, tool->data(), transform);
        if (frames != nullptr) {
            std::copy(transform, transform + 16, frames + 16 * (joints + 1));
        }
    }
    if (endpoint != nullptr) {
        std::copy(transform, transform + 16, endpoint);
    }
    if (active != active_count) {
        throw std::invalid_argument("URDF active joint count does not match q");
    }
}

static py::array_t<double> forward_kinematics_batch_urdf(
    Batch origin_xyz_, Batch origin_rpy_, Batch axis_, JointTypes joint_type_, Batch q_,
    py::object base, py::object tool) {
    const auto origin_xyz = origin_xyz_.request();
    const auto origin_rpy = origin_rpy_.request();
    const auto axis = axis_.request();
    const auto joint_type = joint_type_.request();
    const auto q = q_.request();
    py::ssize_t active_count;
    validate_chain_inputs(origin_xyz, origin_rpy, axis, joint_type, q, &active_count);
    const Transform base_transform = read_transform(base, "base");
    const Transform tool_transform = read_transform(tool, "tool");
    const Transform* tool_ptr = tool.is_none() ? nullptr : &tool_transform;
    const py::ssize_t samples = q.shape[0];
    const py::ssize_t joints = origin_xyz.shape[0];
    py::array_t<double> output(py::array::ShapeContainer(std::vector<py::ssize_t>{samples, 4, 4}));
    auto* output_data = static_cast<double*>(output.request().ptr);
    {
        py::gil_scoped_release release;
        parallel_rows(samples, joints, [&](py::ssize_t begin, py::ssize_t end) {
            std::vector<double> origins(static_cast<std::size_t>(3 * std::max<py::ssize_t>(active_count, 1)));
            std::vector<double> axes(static_cast<std::size_t>(3 * std::max<py::ssize_t>(active_count, 1)));
            for (py::ssize_t sample = begin; sample < end; ++sample) {
                chain_state(
                    static_cast<const double*>(origin_xyz.ptr), static_cast<const double*>(origin_rpy.ptr),
                    static_cast<const double*>(axis.ptr), static_cast<const std::int8_t*>(joint_type.ptr),
                    static_cast<const double*>(q.ptr) + sample * active_count, joints, active_count,
                    base_transform, tool_ptr, output_data + sample * 16, nullptr, origins.data(), axes.data());
            }
        });
    }
    return output;
}

static py::array_t<double> chain_positions_batch_urdf(
    Batch origin_xyz_, Batch origin_rpy_, Batch axis_, JointTypes joint_type_, Batch q_,
    py::object base, py::object tool) {
    const auto origin_xyz = origin_xyz_.request();
    const auto origin_rpy = origin_rpy_.request();
    const auto axis = axis_.request();
    const auto joint_type = joint_type_.request();
    const auto q = q_.request();
    py::ssize_t active_count;
    validate_chain_inputs(origin_xyz, origin_rpy, axis, joint_type, q, &active_count);
    const Transform base_transform = read_transform(base, "base");
    const Transform tool_transform = read_transform(tool, "tool");
    const Transform* tool_ptr = tool.is_none() ? nullptr : &tool_transform;
    const bool has_tool = !tool.is_none();
    const py::ssize_t samples = q.shape[0];
    const py::ssize_t joints = origin_xyz.shape[0];
    const py::ssize_t frame_count = joints + 1 + (has_tool ? 1 : 0);
    py::array_t<double> output(py::array::ShapeContainer(std::vector<py::ssize_t>{samples, frame_count, 3}));
    auto* output_data = static_cast<double*>(output.request().ptr);
    {
        py::gil_scoped_release release;
        parallel_rows(samples, joints, [&](py::ssize_t begin, py::ssize_t end) {
            std::vector<double> frames(static_cast<std::size_t>(16 * frame_count));
            std::vector<double> origins(static_cast<std::size_t>(3 * std::max<py::ssize_t>(active_count, 1)));
            std::vector<double> axes(static_cast<std::size_t>(3 * std::max<py::ssize_t>(active_count, 1)));
            for (py::ssize_t sample = begin; sample < end; ++sample) {
                chain_state(
                    static_cast<const double*>(origin_xyz.ptr), static_cast<const double*>(origin_rpy.ptr),
                    static_cast<const double*>(axis.ptr), static_cast<const std::int8_t*>(joint_type.ptr),
                    static_cast<const double*>(q.ptr) + sample * active_count, joints, active_count,
                    base_transform, tool_ptr, nullptr, frames.data(), origins.data(), axes.data());
                for (py::ssize_t frame = 0; frame < frame_count; ++frame) {
                    output_data[(sample * frame_count + frame) * 3] = frames[16 * frame + 3];
                    output_data[(sample * frame_count + frame) * 3 + 1] = frames[16 * frame + 7];
                    output_data[(sample * frame_count + frame) * 3 + 2] = frames[16 * frame + 11];
                }
            }
        });
    }
    return output;
}

static py::array_t<double> geometric_jacobian_batch_urdf(
    Batch origin_xyz_, Batch origin_rpy_, Batch axis_, JointTypes joint_type_, Batch q_,
    py::object base, py::object tool) {
    const auto origin_xyz = origin_xyz_.request();
    const auto origin_rpy = origin_rpy_.request();
    const auto axis = axis_.request();
    const auto joint_type = joint_type_.request();
    const auto q = q_.request();
    py::ssize_t active_count;
    validate_chain_inputs(origin_xyz, origin_rpy, axis, joint_type, q, &active_count);
    const Transform base_transform = read_transform(base, "base");
    const Transform tool_transform = read_transform(tool, "tool");
    const Transform* tool_ptr = tool.is_none() ? nullptr : &tool_transform;
    const bool has_tool = !tool.is_none();
    std::vector<std::int8_t> active_types;
    const auto* type_data = static_cast<const std::int8_t*>(joint_type.ptr);
    for (py::ssize_t index = 0; index < origin_xyz.shape[0]; ++index) {
        if (type_data[index] != 0) {
            active_types.push_back(type_data[index]);
        }
    }
    const py::ssize_t samples = q.shape[0];
    const py::ssize_t joints = origin_xyz.shape[0];
    py::array_t<double> output(py::array::ShapeContainer(std::vector<py::ssize_t>{samples, 6, active_count}));
    auto* output_data = static_cast<double*>(output.request().ptr);
    {
        py::gil_scoped_release release;
        parallel_rows(samples, joints, [&](py::ssize_t begin, py::ssize_t end) {
            std::vector<double> frames(static_cast<std::size_t>(16 * (joints + 1 + (has_tool ? 1 : 0))));
            std::vector<double> origins(static_cast<std::size_t>(3 * std::max<py::ssize_t>(active_count, 1)));
            std::vector<double> axes(static_cast<std::size_t>(3 * std::max<py::ssize_t>(active_count, 1)));
            double endpoint[16];
            for (py::ssize_t sample = begin; sample < end; ++sample) {
                chain_state(
                    static_cast<const double*>(origin_xyz.ptr), static_cast<const double*>(origin_rpy.ptr),
                    static_cast<const double*>(axis.ptr), static_cast<const std::int8_t*>(joint_type.ptr),
                    static_cast<const double*>(q.ptr) + sample * active_count, joints, active_count,
                    base_transform, tool_ptr, endpoint, frames.data(), origins.data(), axes.data());
                const double end_position[3] = {endpoint[3], endpoint[7], endpoint[11]};
                for (py::ssize_t active = 0; active < active_count; ++active) {
                    const double ax = axes[3 * active];
                    const double ay = axes[3 * active + 1];
                    const double az = axes[3 * active + 2];
                    const double dx = end_position[0] - origins[3 * active];
                    const double dy = end_position[1] - origins[3 * active + 1];
                    const double dz = end_position[2] - origins[3 * active + 2];
                    const py::ssize_t base_index = sample * 6 * active_count + active;
                    if (active == 0) {
                        for (int row = 0; row < 6; ++row) {
                            output_data[sample * 6 * active_count + row * active_count + active] = 0.0;
                        }
                    }
                    const bool revolute = active_types[active] == 1;
                    if (revolute) {
                        output_data[base_index] = ay * dz - az * dy;
                        output_data[sample * 6 * active_count + active_count + active] = az * dx - ax * dz;
                        output_data[sample * 6 * active_count + 2 * active_count + active] = ax * dy - ay * dx;
                        output_data[sample * 6 * active_count + 3 * active_count + active] = ax;
                        output_data[sample * 6 * active_count + 4 * active_count + active] = ay;
                        output_data[sample * 6 * active_count + 5 * active_count + active] = az;
                    } else {
                        output_data[base_index] = ax;
                        output_data[sample * 6 * active_count + active_count + active] = ay;
                        output_data[sample * 6 * active_count + 2 * active_count + active] = az;
                        output_data[sample * 6 * active_count + 3 * active_count + active] = 0.0;
                        output_data[sample * 6 * active_count + 4 * active_count + active] = 0.0;
                        output_data[sample * 6 * active_count + 5 * active_count + active] = 0.0;
                    }
                }
            }
        });
    }
    return output;
}

static void mass_gravity_single(
    const double* origin_xyz,
    const double* origin_rpy,
    const double* axis,
    const std::int8_t* joint_type,
    const double* masses,
    const double* center_of_mass,
    const double* inertias,
    const double* q,
    py::ssize_t joints,
    py::ssize_t active_count,
    const Transform& base,
    const double* gravity,
    double* matrix,
    double* forces) {
    if (matrix != nullptr) {
        std::fill(matrix, matrix + active_count * active_count, 0.0);
    }
    if (forces != nullptr) {
        std::fill(forces, forces + active_count, 0.0);
    }
    std::vector<double> frames(static_cast<std::size_t>(16 * (joints + 1)));
    std::vector<double> origins(static_cast<std::size_t>(3 * std::max<py::ssize_t>(active_count, 1)));
    std::vector<double> axes(static_cast<std::size_t>(3 * std::max<py::ssize_t>(active_count, 1)));
    chain_state(origin_xyz, origin_rpy, axis, joint_type, q, joints, active_count, base, nullptr,
                nullptr, frames.data(), origins.data(), axes.data());
    std::vector<double> linear(static_cast<std::size_t>(3 * std::max<py::ssize_t>(active_count, 1)));
    std::vector<double> angular(static_cast<std::size_t>(3 * std::max<py::ssize_t>(active_count, 1)));
    py::ssize_t active_before = 0;
    for (py::ssize_t body = 0; body < joints; ++body) {
        const double* frame = frames.data() + 16 * (body + 1);
        const double* com = center_of_mass + 3 * body;
        const double position[3] = {
            frame[3] + frame[0] * com[0] + frame[1] * com[1] + frame[2] * com[2],
            frame[7] + frame[4] * com[0] + frame[5] * com[1] + frame[6] * com[2],
            frame[11] + frame[8] * com[0] + frame[9] * com[1] + frame[10] * com[2],
        };
        std::fill(linear.begin(), linear.end(), 0.0);
        std::fill(angular.begin(), angular.end(), 0.0);
        active_before = 0;
        for (py::ssize_t path_joint = 0; path_joint <= body; ++path_joint) {
            if (joint_type[path_joint] == 0) {
                continue;
            }
            const py::ssize_t active = active_before++;
            const double ax = axes[3 * active];
            const double ay = axes[3 * active + 1];
            const double az = axes[3 * active + 2];
            const double dx = position[0] - origins[3 * active];
            const double dy = position[1] - origins[3 * active + 1];
            const double dz = position[2] - origins[3 * active + 2];
            if (joint_type[path_joint] == 1) {
                linear[3 * active] = ay * dz - az * dy;
                linear[3 * active + 1] = az * dx - ax * dz;
                linear[3 * active + 2] = ax * dy - ay * dx;
                angular[3 * active] = ax;
                angular[3 * active + 1] = ay;
                angular[3 * active + 2] = az;
            } else {
                linear[3 * active] = ax;
                linear[3 * active + 1] = ay;
                linear[3 * active + 2] = az;
            }
        }
        if (forces != nullptr && gravity != nullptr) {
            for (py::ssize_t active = 0; active < active_before; ++active) {
                forces[active] -= masses[body] * (
                    linear[3 * active] * gravity[0]
                    + linear[3 * active + 1] * gravity[1]
                    + linear[3 * active + 2] * gravity[2]);
            }
        }
        if (matrix == nullptr) {
            continue;
        }
        double world_inertia[9] = {};
        const double* inertia = inertias + 9 * body;
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
                for (int first = 0; first < 3; ++first) {
                    for (int second = 0; second < 3; ++second) {
                        world_inertia[3 * row + column] += frame[4 * row + first]
                            * inertia[3 * first + second] * frame[4 * column + second];
                    }
                }
            }
        }
        for (py::ssize_t first = 0; first < active_before; ++first) {
            for (py::ssize_t second = 0; second < active_before; ++second) {
                const double linear_dot = linear[3 * first] * linear[3 * second]
                    + linear[3 * first + 1] * linear[3 * second + 1]
                    + linear[3 * first + 2] * linear[3 * second + 2];
                double angular_dot = 0.0;
                for (int row = 0; row < 3; ++row) {
                    for (int column = 0; column < 3; ++column) {
                        angular_dot += angular[3 * first + row] * world_inertia[3 * row + column]
                            * angular[3 * second + column];
                    }
                }
                matrix[first * active_count + second] += masses[body] * linear_dot + angular_dot;
            }
        }
    }
    if (matrix != nullptr) {
        for (py::ssize_t row = 0; row < active_count; ++row) {
            for (py::ssize_t column = row + 1; column < active_count; ++column) {
                const double value = 0.5 * (matrix[row * active_count + column] + matrix[column * active_count + row]);
                matrix[row * active_count + column] = value;
                matrix[column * active_count + row] = value;
            }
        }
    }
}

static void inverse_dynamics_single_urdf(
    const double* origin_xyz,
    const double* origin_rpy,
    const double* axis,
    const std::int8_t* joint_type,
    const double* masses,
    const double* center_of_mass,
    const double* inertias,
    const double* q,
    const double* qd,
    const double* qdd,
    py::ssize_t joints,
    py::ssize_t active_count,
    const Transform& base,
    const double* gravity,
    double* output) {
    std::fill(output, output + active_count, 0.0);
    std::vector<double> frames(static_cast<std::size_t>(16 * (joints + 1)));
    std::vector<double> origins(static_cast<std::size_t>(3 * std::max<py::ssize_t>(active_count, 1)));
    std::vector<double> axes(static_cast<std::size_t>(3 * std::max<py::ssize_t>(active_count, 1)));
    chain_state(origin_xyz, origin_rpy, axis, joint_type, q, joints, active_count, base, nullptr,
                nullptr, frames.data(), origins.data(), axes.data());

    std::vector<double> linear_velocity(static_cast<std::size_t>(3 * joints));
    std::vector<double> angular_velocity(static_cast<std::size_t>(3 * joints));
    std::vector<double> linear_acceleration(static_cast<std::size_t>(3 * joints));
    std::vector<double> angular_acceleration(static_cast<std::size_t>(3 * joints));
    std::vector<double> body_force(static_cast<std::size_t>(3 * joints));
    std::vector<double> body_moment(static_cast<std::size_t>(3 * joints));
    std::vector<double> accumulated_force(static_cast<std::size_t>(3 * joints), 0.0);
    std::vector<double> accumulated_moment(static_cast<std::size_t>(3 * joints), 0.0);
    double base_linear_velocity[3] = {0.0, 0.0, 0.0};
    double base_angular_velocity[3] = {0.0, 0.0, 0.0};
    double base_linear_acceleration[3] = {-gravity[0], -gravity[1], -gravity[2]};
    double base_angular_acceleration[3] = {0.0, 0.0, 0.0};
    py::ssize_t active = 0;
    for (py::ssize_t body = 0; body < joints; ++body) {
        const double* parent_frame = frames.data() + 16 * body;
        const double* child_frame = frames.data() + 16 * (body + 1);
        const double parent_position[3] = {parent_frame[3], parent_frame[7], parent_frame[11]};
        const double child_position[3] = {child_frame[3], child_frame[7], child_frame[11]};
        const double child_offset[3] = {
            child_position[0] - parent_position[0],
            child_position[1] - parent_position[1],
            child_position[2] - parent_position[2],
        };
        const double* parent_velocity = body == 0 ? base_linear_velocity : linear_velocity.data() + 3 * (body - 1);
        const double* parent_angular_velocity = body == 0 ? base_angular_velocity : angular_velocity.data() + 3 * (body - 1);
        const double* parent_acceleration = body == 0 ? base_linear_acceleration : linear_acceleration.data() + 3 * (body - 1);
        const double* parent_angular_acceleration = body == 0 ? base_angular_acceleration : angular_acceleration.data() + 3 * (body - 1);
        double omega_cross_offset[3];
        double alpha_cross_offset[3];
        double omega_cross_omega_cross_offset[3];
        cross3(parent_angular_velocity, child_offset, omega_cross_offset);
        cross3(parent_angular_acceleration, child_offset, alpha_cross_offset);
        cross3(parent_angular_velocity, omega_cross_offset, omega_cross_omega_cross_offset);
        double child_velocity[3];
        add3(parent_velocity, omega_cross_offset, child_velocity);
        double child_acceleration[3];
        double base_acceleration_term[3];
        add3(parent_acceleration, alpha_cross_offset, base_acceleration_term);
        add3(base_acceleration_term, omega_cross_omega_cross_offset, child_acceleration);
        double child_angular_velocity[3] = {
            parent_angular_velocity[0], parent_angular_velocity[1], parent_angular_velocity[2]
        };
        double child_angular_acceleration[3] = {
            parent_angular_acceleration[0], parent_angular_acceleration[1], parent_angular_acceleration[2]
        };
        if (joint_type[body] != 0) {
            const double* joint_axis = axes.data() + 3 * active;
            const double joint_velocity = qd[active];
            const double joint_acceleration = qdd[active];
            if (joint_type[body] == 1) {
                double axis_velocity[3];
                double joint_angular_velocity[3];
                scale3(joint_axis, joint_velocity, joint_angular_velocity);
                cross3(parent_angular_velocity, joint_angular_velocity, axis_velocity);
                add3(child_angular_velocity, joint_angular_velocity, child_angular_velocity);
                double joint_angular_acceleration[3];
                scale3(joint_axis, joint_acceleration, joint_angular_acceleration);
                add3(child_angular_acceleration, joint_angular_acceleration, child_angular_acceleration);
                add3(child_angular_acceleration, axis_velocity, child_angular_acceleration);
            } else {
                double axis_velocity[3];
                cross3(parent_angular_velocity, joint_axis, axis_velocity);
                double twice_axis_velocity[3];
                scale3(axis_velocity, 2.0 * joint_velocity, twice_axis_velocity);
                add3(child_acceleration, twice_axis_velocity, child_acceleration);
                double axis_acceleration[3];
                scale3(joint_axis, joint_acceleration, axis_acceleration);
                add3(child_acceleration, axis_acceleration, child_acceleration);
                double joint_velocity_vector[3];
                scale3(joint_axis, joint_velocity, joint_velocity_vector);
                add3(child_velocity, joint_velocity_vector, child_velocity);
            }
            ++active;
        }
        std::copy(child_velocity, child_velocity + 3, linear_velocity.begin() + 3 * body);
        std::copy(child_angular_velocity, child_angular_velocity + 3, angular_velocity.begin() + 3 * body);
        std::copy(child_acceleration, child_acceleration + 3, linear_acceleration.begin() + 3 * body);
        std::copy(child_angular_acceleration, child_angular_acceleration + 3, angular_acceleration.begin() + 3 * body);

        const double* com = center_of_mass + 3 * body;
        const double center_offset[3] = {
            child_frame[0] * com[0] + child_frame[1] * com[1] + child_frame[2] * com[2],
            child_frame[4] * com[0] + child_frame[5] * com[1] + child_frame[6] * com[2],
            child_frame[8] * com[0] + child_frame[9] * com[1] + child_frame[10] * com[2],
        };
        double alpha_cross_center[3];
        double omega_cross_center[3];
        double omega_cross_omega_center[3];
        cross3(child_angular_acceleration, center_offset, alpha_cross_center);
        cross3(child_angular_velocity, center_offset, omega_cross_center);
        cross3(child_angular_velocity, omega_cross_center, omega_cross_omega_center);
        double center_acceleration[3];
        add3(child_acceleration, alpha_cross_center, center_acceleration);
        add3(center_acceleration, omega_cross_omega_center, center_acceleration);
        double force[3];
        scale3(center_acceleration, masses[body], force);
        std::copy(force, force + 3, body_force.begin() + 3 * body);
        double world_inertia[9] = {};
        const double* inertia = inertias + 9 * body;
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
                for (int first = 0; first < 3; ++first) {
                    for (int second = 0; second < 3; ++second) {
                        world_inertia[3 * row + column] += child_frame[4 * row + first]
                            * inertia[3 * first + second] * child_frame[4 * column + second];
                    }
                }
            }
        }
        double inertia_omega[3] = {0.0, 0.0, 0.0};
        double inertia_alpha[3] = {0.0, 0.0, 0.0};
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
                inertia_omega[row] += world_inertia[3 * row + column] * child_angular_velocity[column];
                inertia_alpha[row] += world_inertia[3 * row + column] * child_angular_acceleration[column];
            }
        }
        double omega_cross_inertia_omega[3];
        double center_cross_force[3];
        cross3(child_angular_velocity, inertia_omega, omega_cross_inertia_omega);
        cross3(center_offset, force, center_cross_force);
        double moment[3];
        add3(inertia_alpha, omega_cross_inertia_omega, moment);
        add3(moment, center_cross_force, moment);
        std::copy(moment, moment + 3, body_moment.begin() + 3 * body);
    }

    active = active_count;
    for (py::ssize_t body = joints - 1; body >= 0; --body) {
        double total_force[3] = {
            body_force[3 * body] + accumulated_force[3 * body],
            body_force[3 * body + 1] + accumulated_force[3 * body + 1],
            body_force[3 * body + 2] + accumulated_force[3 * body + 2],
        };
        double total_moment[3] = {
            body_moment[3 * body] + accumulated_moment[3 * body],
            body_moment[3 * body + 1] + accumulated_moment[3 * body + 1],
            body_moment[3 * body + 2] + accumulated_moment[3 * body + 2],
        };
        if (joint_type[body] != 0) {
            --active;
            const double* joint_axis = axes.data() + 3 * active;
            if (joint_type[body] == 1) {
                output[active] = joint_axis[0] * total_moment[0]
                    + joint_axis[1] * total_moment[1] + joint_axis[2] * total_moment[2];
            } else {
                output[active] = joint_axis[0] * total_force[0]
                    + joint_axis[1] * total_force[1] + joint_axis[2] * total_force[2];
            }
        }
        if (body > 0) {
            const double* child_frame = frames.data() + 16 * (body + 1);
            const double* parent_frame = frames.data() + 16 * body;
            const double offset[3] = {
                child_frame[3] - parent_frame[3],
                child_frame[7] - parent_frame[7],
                child_frame[11] - parent_frame[11],
            };
            double offset_cross_force[3];
            cross3(offset, total_force, offset_cross_force);
            accumulated_force[3 * (body - 1)] += total_force[0];
            accumulated_force[3 * (body - 1) + 1] += total_force[1];
            accumulated_force[3 * (body - 1) + 2] += total_force[2];
            accumulated_moment[3 * (body - 1)] += total_moment[0] + offset_cross_force[0];
            accumulated_moment[3 * (body - 1) + 1] += total_moment[1] + offset_cross_force[1];
            accumulated_moment[3 * (body - 1) + 2] += total_moment[2] + offset_cross_force[2];
        }
    }
}

static void validate_dynamics_inputs(
    const py::buffer_info& origin_xyz,
    const py::buffer_info& origin_rpy,
    const py::buffer_info& axis,
    const py::buffer_info& joint_type,
    const py::buffer_info& masses,
    const py::buffer_info& center_of_mass,
    const py::buffer_info& inertias,
    const py::buffer_info& q,
    const py::buffer_info& qd,
    const py::buffer_info& gravity,
    py::ssize_t* active_count) {
    validate_chain_inputs(origin_xyz, origin_rpy, axis, joint_type, q, active_count);
    const py::ssize_t joints = origin_xyz.shape[0];
    if (masses.ndim != 1 || masses.shape[0] != joints) {
        throw std::invalid_argument("masses must have shape (n_path_joints,)");
    }
    if (center_of_mass.ndim != 2 || center_of_mass.shape[0] != joints || center_of_mass.shape[1] != 3) {
        throw std::invalid_argument("center_of_mass must have shape (n_path_joints, 3)");
    }
    if (inertias.ndim != 3 || inertias.shape[0] != joints || inertias.shape[1] != 3 || inertias.shape[2] != 3) {
        throw std::invalid_argument("inertias must have shape (n_path_joints, 3, 3)");
    }
    if (qd.ndim != 2 || qd.shape[0] != q.shape[0] || qd.shape[1] != *active_count) {
        throw std::invalid_argument("qd must have the same shape as q");
    }
    if (gravity.ndim != 1 || gravity.shape[0] != 3) {
        throw std::invalid_argument("gravity must have shape (3,)");
    }
    require_finite(static_cast<const double*>(masses.ptr), masses.size, "masses");
    require_finite(static_cast<const double*>(center_of_mass.ptr), center_of_mass.size, "center_of_mass");
    require_finite(static_cast<const double*>(inertias.ptr), inertias.size, "inertias");
    require_finite(static_cast<const double*>(qd.ptr), qd.size, "qd");
    require_finite(static_cast<const double*>(gravity.ptr), gravity.size, "gravity");
}

static py::tuple spatial_dynamics_batch_urdf(
    Batch origin_xyz_, Batch origin_rpy_, Batch axis_, JointTypes joint_type_,
    Vector masses_, Batch center_of_mass_, Batch inertias_, Batch q_, Batch qd_,
    Vector gravity_, double epsilon, bool compute_coriolis, py::object base) {
    const auto origin_xyz = origin_xyz_.request();
    const auto origin_rpy = origin_rpy_.request();
    const auto axis = axis_.request();
    const auto joint_type = joint_type_.request();
    const auto masses = masses_.request();
    const auto center_of_mass = center_of_mass_.request();
    const auto inertias = inertias_.request();
    const auto q = q_.request();
    const auto qd = qd_.request();
    const auto gravity = gravity_.request();
    py::ssize_t active_count;
    validate_dynamics_inputs(
        origin_xyz, origin_rpy, axis, joint_type, masses, center_of_mass, inertias,
        q, qd, gravity, &active_count);
    if (!std::isfinite(epsilon) || epsilon <= 0.0) {
        throw std::invalid_argument("epsilon must be a positive finite value");
    }
    const Transform base_transform = read_transform(base, "base");
    const py::ssize_t samples = q.shape[0];
    const py::ssize_t joints = origin_xyz.shape[0];
    py::array_t<double> matrices(py::array::ShapeContainer(std::vector<py::ssize_t>{samples, active_count, active_count}));
    py::array_t<double> forces(py::array::ShapeContainer(std::vector<py::ssize_t>{samples, active_count}));
    py::array_t<double> coriolis(py::array::ShapeContainer(std::vector<py::ssize_t>{samples, active_count}));
    auto* matrix_data = static_cast<double*>(matrices.request().ptr);
    auto* force_data = static_cast<double*>(forces.request().ptr);
    auto* coriolis_data = static_cast<double*>(coriolis.request().ptr);
    const auto* origin_xyz_data = static_cast<const double*>(origin_xyz.ptr);
    const auto* origin_rpy_data = static_cast<const double*>(origin_rpy.ptr);
    const auto* axis_data = static_cast<const double*>(axis.ptr);
    const auto* joint_type_data = static_cast<const std::int8_t*>(joint_type.ptr);
    const auto* masses_data = static_cast<const double*>(masses.ptr);
    const auto* center_of_mass_data = static_cast<const double*>(center_of_mass.ptr);
    const auto* inertias_data = static_cast<const double*>(inertias.ptr);
    const auto* q_data = static_cast<const double*>(q.ptr);
    const auto* qd_data = static_cast<const double*>(qd.ptr);
    const auto* gravity_data = static_cast<const double*>(gravity.ptr);
    {
        py::gil_scoped_release release;
        parallel_rows(samples, joints * std::max<py::ssize_t>(active_count, 1), [&](py::ssize_t begin, py::ssize_t end) {
            for (py::ssize_t sample = begin; sample < end; ++sample) {
                mass_gravity_single(
                    origin_xyz_data, origin_rpy_data, axis_data, joint_type_data,
                    masses_data, center_of_mass_data, inertias_data,
                    q_data + sample * active_count, joints, active_count,
                    base_transform, gravity_data,
                    matrix_data + sample * active_count * active_count,
                    force_data + sample * active_count);
                std::fill(coriolis_data + sample * active_count, coriolis_data + (sample + 1) * active_count, 0.0);
                if (!compute_coriolis || active_count == 0) {
                    continue;
                }
                std::vector<double> derivatives(static_cast<std::size_t>(active_count * active_count * active_count));
                std::vector<double> plus(static_cast<std::size_t>(active_count * active_count));
                std::vector<double> minus(static_cast<std::size_t>(active_count * active_count));
                std::vector<double> q_plus(static_cast<std::size_t>(active_count));
                std::vector<double> q_minus(static_cast<std::size_t>(active_count));
                const double* values = q_data + sample * active_count;
                for (py::ssize_t coordinate = 0; coordinate < active_count; ++coordinate) {
                    std::copy(values, values + active_count, q_plus.begin());
                    std::copy(values, values + active_count, q_minus.begin());
                    q_plus[coordinate] += epsilon;
                    q_minus[coordinate] -= epsilon;
                    mass_gravity_single(
                        origin_xyz_data, origin_rpy_data, axis_data, joint_type_data,
                        masses_data, center_of_mass_data, inertias_data, q_plus.data(),
                        joints, active_count, base_transform, nullptr, plus.data(), nullptr);
                    mass_gravity_single(
                        origin_xyz_data, origin_rpy_data, axis_data, joint_type_data,
                        masses_data, center_of_mass_data, inertias_data, q_minus.data(),
                        joints, active_count, base_transform, nullptr, minus.data(), nullptr);
                    for (py::ssize_t row = 0; row < active_count; ++row) {
                        for (py::ssize_t column = 0; column < active_count; ++column) {
                            derivatives[coordinate * active_count * active_count + row * active_count + column] =
                                (plus[row * active_count + column] - minus[row * active_count + column])
                                / (2.0 * epsilon);
                        }
                    }
                }
                const double* velocities = qd_data + sample * active_count;
                for (py::ssize_t row = 0; row < active_count; ++row) {
                    for (py::ssize_t first = 0; first < active_count; ++first) {
                        for (py::ssize_t second = 0; second < active_count; ++second) {
                            const double first_term = derivatives[second * active_count * active_count + row * active_count + first];
                            const double second_term = derivatives[first * active_count * active_count + row * active_count + second];
                            const double third_term = derivatives[row * active_count * active_count + first * active_count + second];
                            coriolis_data[sample * active_count + row] += 0.5
                                * (first_term + second_term - third_term)
                                * velocities[first] * velocities[second];
                        }
                    }
                }
            }
        });
    }
    return py::make_tuple(matrices, forces, coriolis);
}

static py::array_t<double> inverse_dynamics_batch_urdf(
    Batch origin_xyz_, Batch origin_rpy_, Batch axis_, JointTypes joint_type_,
    Vector masses_, Batch center_of_mass_, Batch inertias_, Batch q_, Batch qd_, Batch qdd_,
    Vector gravity_, py::object base) {
    const auto origin_xyz = origin_xyz_.request();
    const auto origin_rpy = origin_rpy_.request();
    const auto axis = axis_.request();
    const auto joint_type = joint_type_.request();
    const auto masses = masses_.request();
    const auto center_of_mass = center_of_mass_.request();
    const auto inertias = inertias_.request();
    const auto q = q_.request();
    const auto qd = qd_.request();
    const auto qdd = qdd_.request();
    const auto gravity = gravity_.request();
    py::ssize_t active_count;
    validate_dynamics_inputs(
        origin_xyz, origin_rpy, axis, joint_type, masses, center_of_mass, inertias,
        q, qd, gravity, &active_count);
    if (qdd.ndim != 2 || qdd.shape[0] != q.shape[0] || qdd.shape[1] != active_count) {
        throw std::invalid_argument("qdd must have the same shape as q");
    }
    require_finite(static_cast<const double*>(qdd.ptr), qdd.size, "qdd");
    const Transform base_transform = read_transform(base, "base");
    const py::ssize_t samples = q.shape[0];
    const py::ssize_t joints = origin_xyz.shape[0];
    py::array_t<double> output(py::array::ShapeContainer(std::vector<py::ssize_t>{samples, active_count}));
    auto* output_data = static_cast<double*>(output.request().ptr);
    {
        py::gil_scoped_release release;
        parallel_rows(samples, joints * std::max<py::ssize_t>(active_count, 1), [&](py::ssize_t begin, py::ssize_t end) {
            for (py::ssize_t sample = begin; sample < end; ++sample) {
                inverse_dynamics_single_urdf(
                    static_cast<const double*>(origin_xyz.ptr), static_cast<const double*>(origin_rpy.ptr),
                    static_cast<const double*>(axis.ptr), static_cast<const std::int8_t*>(joint_type.ptr),
                    static_cast<const double*>(masses.ptr), static_cast<const double*>(center_of_mass.ptr),
                    static_cast<const double*>(inertias.ptr), static_cast<const double*>(q.ptr) + sample * active_count,
                    static_cast<const double*>(qd.ptr) + sample * active_count,
                    static_cast<const double*>(qdd.ptr) + sample * active_count,
                    joints, active_count, base_transform, static_cast<const double*>(gravity.ptr),
                    output_data + sample * active_count);
            }
        });
    }
    return output;
}

PYBIND11_MODULE(_spatial_cpp, module) {
    module.doc() = "C++ accelerated general URDF spatial kinematics and dynamics";
    module.def(
        "forward_kinematics_batch_urdf", &forward_kinematics_batch_urdf,
        py::arg("origin_xyz"), py::arg("origin_rpy"), py::arg("axis"), py::arg("joint_type"),
        py::arg("q"), py::arg("base") = py::none(), py::arg("tool") = py::none());
    module.def(
        "chain_positions_batch_urdf", &chain_positions_batch_urdf,
        py::arg("origin_xyz"), py::arg("origin_rpy"), py::arg("axis"), py::arg("joint_type"),
        py::arg("q"), py::arg("base") = py::none(), py::arg("tool") = py::none());
    module.def(
        "geometric_jacobian_batch_urdf", &geometric_jacobian_batch_urdf,
        py::arg("origin_xyz"), py::arg("origin_rpy"), py::arg("axis"), py::arg("joint_type"),
        py::arg("q"), py::arg("base") = py::none(), py::arg("tool") = py::none());
    module.def(
        "spatial_dynamics_batch_urdf", &spatial_dynamics_batch_urdf,
        py::arg("origin_xyz"), py::arg("origin_rpy"), py::arg("axis"), py::arg("joint_type"),
        py::arg("masses"), py::arg("center_of_mass"), py::arg("inertias"), py::arg("q"),
        py::arg("qd"), py::arg("gravity"), py::arg("epsilon") = 1e-6,
        py::arg("compute_coriolis") = false, py::arg("base") = py::none());
    module.def(
        "inverse_dynamics_batch_urdf", &inverse_dynamics_batch_urdf,
        py::arg("origin_xyz"), py::arg("origin_rpy"), py::arg("axis"), py::arg("joint_type"),
        py::arg("masses"), py::arg("center_of_mass"), py::arg("inertias"), py::arg("q"),
        py::arg("qd"), py::arg("qdd"), py::arg("gravity"), py::arg("base") = py::none());
}
