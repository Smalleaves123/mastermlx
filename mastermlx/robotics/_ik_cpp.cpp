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
using Matrix = py::array_t<double, py::array::c_style | py::array::forcecast>;
using Transform = std::array<double, 16>;

static inline double clamp_value(double value, double lower, double upper) {
    return std::max(lower, std::min(value, upper));
}

static Transform read_transform(py::handle value, const char* name) {
    if (value.is_none()) {
        return {1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0};
    }
    Vector array = Vector::ensure(value);
    if (!array) {
        throw std::invalid_argument(std::string(name) + " must be array-like");
    }
    const auto info = array.request();
    if (info.ndim != 2 || info.shape[0] != 4 || info.shape[1] != 4) {
        throw std::invalid_argument(std::string(name) + " must have shape (4, 4)");
    }
    Transform result{};
    const auto* data = static_cast<const double*>(info.ptr);
    std::copy(data, data + 16, result.begin());
    return result;
}

static inline void matmul4(const double* left, const double* right, double* output) {
    double value[16];
    for (int row = 0; row < 4; ++row) {
        for (int column = 0; column < 4; ++column) {
            value[row * 4 + column] =
                left[row * 4] * right[column]
                + left[row * 4 + 1] * right[4 + column]
                + left[row * 4 + 2] * right[8 + column]
                + left[row * 4 + 3] * right[12 + column];
        }
    }
    std::copy(value, value + 16, output);
}

static void chain_state(
    const double* a,
    const double* alpha,
    const double* d,
    const double* theta,
    const std::int8_t* joint_type,
    const double* offset,
    const double* q,
    py::ssize_t joints,
    const Transform& base,
    const Transform* tool,
    double* transform_out,
    double* origins,
    double* axes) {
    double transform[16];
    double next[16];
    std::copy(base.begin(), base.end(), transform);
    for (py::ssize_t index = 0; index < joints; ++index) {
        origins[3 * index] = transform[3];
        origins[3 * index + 1] = transform[7];
        origins[3 * index + 2] = transform[11];
        axes[3 * index] = transform[2];
        axes[3 * index + 1] = transform[6];
        axes[3 * index + 2] = transform[10];

        double joint_d = d[index];
        double joint_theta = theta[index];
        if (joint_type[index] != 0) {
            joint_theta += q[index] + offset[index];
        } else {
            joint_d += q[index] + offset[index];
        }
        const double ct = std::cos(joint_theta);
        const double st = std::sin(joint_theta);
        const double ca = std::cos(alpha[index]);
        const double sa = std::sin(alpha[index]);
        const double link[16] = {
            ct, -st * ca, st * sa, a[index] * ct,
            st, ct * ca, -ct * sa, a[index] * st,
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
    std::copy(transform, transform + 16, transform_out);
}

static bool solve_symmetric_3x3(const double* matrix, const double* rhs, double* solution) {
    double augmented[3][4];
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            augmented[row][column] = matrix[row * 3 + column];
        }
        augmented[row][3] = rhs[row];
    }
    for (int column = 0; column < 3; ++column) {
        int pivot = column;
        for (int row = column + 1; row < 3; ++row) {
            if (std::abs(augmented[row][column]) > std::abs(augmented[pivot][column])) {
                pivot = row;
            }
        }
        if (std::abs(augmented[pivot][column]) <= 1e-15) {
            return false;
        }
        if (pivot != column) {
            for (int item = column; item < 4; ++item) {
                std::swap(augmented[column][item], augmented[pivot][item]);
            }
        }
        const double divisor = augmented[column][column];
        for (int item = column; item < 4; ++item) {
            augmented[column][item] /= divisor;
        }
        for (int row = 0; row < 3; ++row) {
            if (row == column) {
                continue;
            }
            const double factor = augmented[row][column];
            for (int item = column; item < 4; ++item) {
                augmented[row][item] -= factor * augmented[column][item];
            }
        }
    }
    for (int row = 0; row < 3; ++row) {
        solution[row] = augmented[row][3];
    }
    return true;
}

static void solve_position_target(
    const double* a,
    const double* alpha,
    const double* d,
    const double* theta,
    const std::int8_t* joint_type,
    const double* offset,
    const double* target,
    const double* seed,
    py::ssize_t joints,
    const Transform& base,
    const Transform* tool,
    const double* limits,
    int max_iter,
    double tol,
    double damping,
    double step_size,
    double* output) {
    std::vector<double> q(seed, seed + joints);
    std::vector<double> origins(static_cast<std::size_t>(3 * joints));
    std::vector<double> axes(static_cast<std::size_t>(3 * joints));
    double transform[16];
    std::vector<double> jacobian(static_cast<std::size_t>(3 * joints));
    for (int iteration = 0; iteration < max_iter; ++iteration) {
        chain_state(
            a, alpha, d, theta, joint_type, offset, q.data(), joints, base, tool,
            transform, origins.data(), axes.data());
        const double error[3] = {
            target[0] - transform[3],
            target[1] - transform[7],
            target[2] - transform[11],
        };
        const double error_norm = std::sqrt(error[0] * error[0] + error[1] * error[1] + error[2] * error[2]);
        if (error_norm <= tol) {
            break;
        }
        double gram[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        const double end[3] = {transform[3], transform[7], transform[11]};
        for (py::ssize_t joint = 0; joint < joints; ++joint) {
            const double dx = end[0] - origins[3 * joint];
            const double dy = end[1] - origins[3 * joint + 1];
            const double dz = end[2] - origins[3 * joint + 2];
            const double ax = axes[3 * joint];
            const double ay = axes[3 * joint + 1];
            const double az = axes[3 * joint + 2];
            double jx;
            double jy;
            double jz;
            if (joint_type[joint] != 0) {
                jx = ay * dz - az * dy;
                jy = az * dx - ax * dz;
                jz = ax * dy - ay * dx;
            } else {
                jx = ax;
                jy = ay;
                jz = az;
            }
            jacobian[3 * joint] = jx;
            jacobian[3 * joint + 1] = jy;
            jacobian[3 * joint + 2] = jz;
            gram[0] += jx * jx;
            gram[1] += jx * jy;
            gram[2] += jx * jz;
            gram[4] += jy * jy;
            gram[5] += jy * jz;
            gram[8] += jz * jz;
        }
        gram[3] = gram[1];
        gram[6] = gram[2];
        gram[7] = gram[5];
        const double damping_sq = damping * damping;
        gram[0] += damping_sq;
        gram[4] += damping_sq;
        gram[8] += damping_sq;
        double solve[3];
        if (!solve_symmetric_3x3(gram, error, solve)) {
            break;
        }
        for (py::ssize_t joint = 0; joint < joints; ++joint) {
            const double* column = jacobian.data() + 3 * joint;
            q[joint] += step_size * (column[0] * solve[0] + column[1] * solve[1] + column[2] * solve[2]);
            if (limits != nullptr) {
                q[joint] = clamp_value(q[joint], limits[2 * joint], limits[2 * joint + 1]);
            }
        }
    }
    std::copy(q.begin(), q.end(), output);
}

static py::array_t<double> inverse_kinematics_position_batch_dh(
    Vector a_, Vector alpha_, Vector d_, Vector theta_, JointTypes joint_type_, Vector offset_,
    Matrix targets_, py::object seeds, py::object base, py::object tool, py::object joint_limits,
    int max_iter, double tol, double damping, double step_size, bool warm_start) {
    const auto a = a_.request();
    const auto alpha = alpha_.request();
    const auto d = d_.request();
    const auto theta = theta_.request();
    const auto joint_type = joint_type_.request();
    const auto offset = offset_.request();
    const auto targets = targets_.request();
    const py::ssize_t joints = a.shape[0];
    if (a.ndim != 1 || alpha.ndim != 1 || d.ndim != 1 || theta.ndim != 1
        || joint_type.ndim != 1 || offset.ndim != 1
        || alpha.shape[0] != joints || d.shape[0] != joints || theta.shape[0] != joints
        || joint_type.shape[0] != joints || offset.shape[0] != joints) {
        throw std::invalid_argument("DH parameter arrays must have the same length");
    }
    if (targets.ndim != 2 || targets.shape[1] != 3 || targets.shape[0] < 1) {
        throw std::invalid_argument("targets must have shape (n_targets, 3)");
    }
    const Transform base_transform = read_transform(base, "base");
    const bool has_tool = !tool.is_none();
    const Transform tool_transform = read_transform(tool, "tool");

    Vector seed_array;
    int seed_mode = 0;
    if (!seeds.is_none()) {
        seed_array = Vector::ensure(seeds);
        if (!seed_array) {
            throw std::invalid_argument("joint_values must be array-like");
        }
        const auto seed_info = seed_array.request();
        if (seed_info.ndim == 1 && seed_info.shape[0] == joints) {
            seed_mode = 1;
        } else if (seed_info.ndim == 2 && seed_info.shape[0] == targets.shape[0] && seed_info.shape[1] == joints) {
            seed_mode = 2;
            if (warm_start) {
                throw std::invalid_argument("per-target joint_values require warm_start=False");
            }
        } else {
            throw std::invalid_argument("joint_values must be a vector or a per-target matrix");
        }
    }

    Vector limits_array;
    const double* limits = nullptr;
    if (!joint_limits.is_none()) {
        limits_array = Vector::ensure(joint_limits);
        if (!limits_array) {
            throw std::invalid_argument("joint_limits must be array-like");
        }
        const auto limit_info = limits_array.request();
        if (limit_info.ndim != 2 || limit_info.shape[0] != joints || limit_info.shape[1] != 2) {
            throw std::invalid_argument("joint_limits must have shape (n_joints, 2)");
        }
        limits = static_cast<const double*>(limit_info.ptr);
    }

    const py::ssize_t n_targets = targets.shape[0];
    const auto* target_data = static_cast<const double*>(targets.ptr);
    const auto* seed_data = seed_mode == 0 ? nullptr : static_cast<const double*>(seed_array.request().ptr);
    const auto* a_data = static_cast<const double*>(a.ptr);
    const auto* alpha_data = static_cast<const double*>(alpha.ptr);
    const auto* d_data = static_cast<const double*>(d.ptr);
    const auto* theta_data = static_cast<const double*>(theta.ptr);
    const auto* type_data = static_cast<const std::int8_t*>(joint_type.ptr);
    const auto* offset_data = static_cast<const double*>(offset.ptr);
    py::array_t<double> output(
        py::array::ShapeContainer(std::vector<py::ssize_t>{n_targets, joints}));
    auto* output_data = static_cast<double*>(output.request().ptr);

    std::vector<double> default_seed(static_cast<std::size_t>(joints), 0.0);
    if (limits != nullptr) {
        for (py::ssize_t joint = 0; joint < joints; ++joint) {
            default_seed[static_cast<std::size_t>(joint)] = 0.5 * (limits[2 * joint] + limits[2 * joint + 1]);
        }
    }
    {
        py::gil_scoped_release release;
        std::vector<double> previous = default_seed;
        for (py::ssize_t target_index = 0; target_index < n_targets; ++target_index) {
            std::vector<double> seed = default_seed;
            if (seed_mode == 1) {
                if (warm_start && target_index > 0) {
                    seed = previous;
                } else {
                    seed.assign(seed_data, seed_data + joints);
                }
            } else if (seed_mode == 2) {
                seed.assign(seed_data + target_index * joints, seed_data + (target_index + 1) * joints);
            } else if (warm_start && target_index > 0) {
                seed = previous;
            }
            solve_position_target(
                a_data, alpha_data, d_data, theta_data, type_data, offset_data,
                target_data + target_index * 3, seed.data(), joints, base_transform,
                has_tool ? &tool_transform : nullptr, limits, max_iter, tol, damping,
                step_size, output_data + target_index * joints);
            previous.assign(output_data + target_index * joints, output_data + (target_index + 1) * joints);
        }
    }
    return output;
}

PYBIND11_MODULE(_ik_cpp, m) {
    m.doc() = "C++ accelerated batch position inverse kinematics";
    m.def(
        "inverse_kinematics_position_batch_dh", &inverse_kinematics_position_batch_dh,
        py::arg("a"), py::arg("alpha"), py::arg("d"), py::arg("theta"),
        py::arg("joint_type"), py::arg("offset"), py::arg("targets"),
        py::arg("seeds") = py::none(), py::arg("base") = py::none(), py::arg("tool") = py::none(),
        py::arg("joint_limits") = py::none(), py::arg("max_iter") = 100,
        py::arg("tol") = 1e-6, py::arg("damping") = 1e-4, py::arg("step_size") = 1.0,
        py::arg("warm_start") = true);
}
