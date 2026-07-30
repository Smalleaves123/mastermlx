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

static void require_finite(const double* values, py::ssize_t size, const char* name) {
    for (py::ssize_t index = 0; index < size; ++index) {
        if (!std::isfinite(values[index])) {
            throw std::invalid_argument(std::string(name) + " must contain only finite values");
        }
    }
}

static Transform read_transform(py::handle value) {
    Transform result = {1.0, 0.0, 0.0, 0.0,
                        0.0, 1.0, 0.0, 0.0,
                        0.0, 0.0, 1.0, 0.0,
                        0.0, 0.0, 0.0, 1.0};
    if (value.is_none()) {
        return result;
    }
    Vector array = Vector::ensure(value);
    if (!array) {
        throw std::invalid_argument("base must be array-like");
    }
    const auto info = array.request();
    if (info.ndim != 2 || info.shape[0] != 4 || info.shape[1] != 4) {
        throw std::invalid_argument("base must have shape (4, 4)");
    }
    const auto* data = static_cast<const double*>(info.ptr);
    require_finite(data, 16, "base");
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

static void validate_inputs(
    const py::buffer_info& a,
    const py::buffer_info& alpha,
    const py::buffer_info& d,
    const py::buffer_info& theta,
    const py::buffer_info& joint_type,
    const py::buffer_info& offset,
    const py::buffer_info& masses,
    const py::buffer_info& center_of_mass,
    const py::buffer_info& inertias,
    const py::buffer_info& q) {
    const py::ssize_t joints = a.shape[0];
    if (a.ndim != 1 || alpha.ndim != 1 || d.ndim != 1 || theta.ndim != 1
        || joint_type.ndim != 1 || offset.ndim != 1 || masses.ndim != 1
        || alpha.shape[0] != joints || d.shape[0] != joints || theta.shape[0] != joints
        || joint_type.shape[0] != joints || offset.shape[0] != joints || masses.shape[0] != joints) {
        throw std::invalid_argument("DH and mass arrays must be 1D with the same length");
    }
    if (center_of_mass.ndim != 2 || center_of_mass.shape[0] != joints || center_of_mass.shape[1] != 3) {
        throw std::invalid_argument("center_of_mass must have shape (n_joints, 3)");
    }
    if (inertias.ndim != 3 || inertias.shape[0] != joints || inertias.shape[1] != 3 || inertias.shape[2] != 3) {
        throw std::invalid_argument("inertias must have shape (n_joints, 3, 3)");
    }
    if (q.ndim != 2 || q.shape[1] != joints || q.shape[0] < 1) {
        throw std::invalid_argument("q must have shape (n_samples, n_joints)");
    }
    require_finite(static_cast<const double*>(a.ptr), joints, "a");
    require_finite(static_cast<const double*>(alpha.ptr), joints, "alpha");
    require_finite(static_cast<const double*>(d.ptr), joints, "d");
    require_finite(static_cast<const double*>(theta.ptr), joints, "theta");
    require_finite(static_cast<const double*>(offset.ptr), joints, "offset");
    require_finite(static_cast<const double*>(masses.ptr), joints, "masses");
    require_finite(static_cast<const double*>(center_of_mass.ptr), center_of_mass.size, "center_of_mass");
    require_finite(static_cast<const double*>(inertias.ptr), inertias.size, "inertias");
    require_finite(static_cast<const double*>(q.ptr), q.size, "q");
}

static py::array_t<double> output_array(
    py::object requested, const std::vector<py::ssize_t>& shape, const char* name) {
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

static void mass_and_gravity_single(
    const double* a, const double* alpha, const double* d, const double* theta,
    const std::int8_t* joint_type, const double* offset, const double* masses,
    const double* center_of_mass, const double* inertias, const double* q,
    py::ssize_t joints, const Transform& base, const double* gravity,
    double* matrix, double* forces) {
    if (matrix != nullptr) {
        std::fill(matrix, matrix + joints * joints, 0.0);
    }
    if (forces != nullptr) {
        std::fill(forces, forces + joints, 0.0);
    }
    std::vector<double> frames(static_cast<std::size_t>(16 * joints));
    std::vector<double> origins(static_cast<std::size_t>(3 * joints));
    std::vector<double> axes(static_cast<std::size_t>(3 * joints));
    double transform[16];
    std::copy(base.begin(), base.end(), transform);

    for (py::ssize_t joint = 0; joint < joints; ++joint) {
        origins[3 * joint] = transform[3];
        origins[3 * joint + 1] = transform[7];
        origins[3 * joint + 2] = transform[11];
        axes[3 * joint] = transform[2];
        axes[3 * joint + 1] = transform[6];
        axes[3 * joint + 2] = transform[10];
        double joint_d = d[joint];
        double joint_theta = theta[joint];
        if (joint_type[joint] != 0) {
            joint_theta += q[joint] + offset[joint];
        } else {
            joint_d += q[joint] + offset[joint];
        }
        const double ct = std::cos(joint_theta);
        const double st = std::sin(joint_theta);
        const double ca = std::cos(alpha[joint]);
        const double sa = std::sin(alpha[joint]);
        const double link[16] = {
            ct, -st * ca, st * sa, a[joint] * ct,
            st, ct * ca, -ct * sa, a[joint] * st,
            0.0, sa, ca, joint_d,
            0.0, 0.0, 0.0, 1.0,
        };
        double next[16];
        matmul4(transform, link, next);
        std::copy(next, next + 16, transform);
        std::copy(transform, transform + 16, frames.begin() + 16 * joint);
    }

    std::vector<double> linear(static_cast<std::size_t>(3 * joints));
    for (py::ssize_t body = 0; body < joints; ++body) {
        const double* frame = frames.data() + 16 * body;
        const double* com = center_of_mass + 3 * body;
        const double position[3] = {
            frame[3] + frame[0] * com[0] + frame[1] * com[1] + frame[2] * com[2],
            frame[7] + frame[4] * com[0] + frame[5] * com[1] + frame[6] * com[2],
            frame[11] + frame[8] * com[0] + frame[9] * com[1] + frame[10] * com[2],
        };
        std::fill(linear.begin(), linear.end(), 0.0);
        for (py::ssize_t joint = 0; joint <= body; ++joint) {
            const double ax = axes[3 * joint];
            const double ay = axes[3 * joint + 1];
            const double az = axes[3 * joint + 2];
            if (joint_type[joint] != 0) {
                const double dx = position[0] - origins[3 * joint];
                const double dy = position[1] - origins[3 * joint + 1];
                const double dz = position[2] - origins[3 * joint + 2];
                linear[3 * joint] = ay * dz - az * dy;
                linear[3 * joint + 1] = az * dx - ax * dz;
                linear[3 * joint + 2] = ax * dy - ay * dx;
            } else {
                linear[3 * joint] = ax;
                linear[3 * joint + 1] = ay;
                linear[3 * joint + 2] = az;
            }
        }
        if (forces != nullptr && gravity != nullptr) {
            for (py::ssize_t joint = 0; joint <= body; ++joint) {
                const double projection = linear[3 * joint] * gravity[0]
                    + linear[3 * joint + 1] * gravity[1]
                    + linear[3 * joint + 2] * gravity[2];
                forces[joint] -= masses[body] * projection;
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
        for (py::ssize_t first = 0; first <= body; ++first) {
            for (py::ssize_t second = 0; second <= body; ++second) {
                const double linear_dot = linear[3 * first] * linear[3 * second]
                    + linear[3 * first + 1] * linear[3 * second + 1]
                    + linear[3 * first + 2] * linear[3 * second + 2];
                double angular_dot = 0.0;
                if (joint_type[first] != 0 && joint_type[second] != 0) {
                    const double* first_axis = axes.data() + 3 * first;
                    const double* second_axis = axes.data() + 3 * second;
                    for (int row = 0; row < 3; ++row) {
                        for (int column = 0; column < 3; ++column) {
                            angular_dot += first_axis[row] * world_inertia[3 * row + column]
                                * second_axis[column];
                        }
                    }
                }
                matrix[first * joints + second] += masses[body] * linear_dot + angular_dot;
            }
        }
    }
    if (matrix != nullptr) {
        for (py::ssize_t row = 0; row < joints; ++row) {
            for (py::ssize_t column = row + 1; column < joints; ++column) {
                const double value = 0.5 * (matrix[row * joints + column] + matrix[column * joints + row]);
                matrix[row * joints + column] = value;
                matrix[column * joints + row] = value;
            }
        }
    }
}

static py::array_t<double> mass_matrix_batch_dh(
    Vector a_, Vector alpha_, Vector d_, Vector theta_, JointTypes joint_type_, Vector offset_,
    Vector masses_, Batch center_of_mass_, Batch inertias_, Batch q_, py::object base,
    py::object requested_output) {
    const auto a = a_.request();
    const auto alpha = alpha_.request();
    const auto d = d_.request();
    const auto theta = theta_.request();
    const auto joint_type = joint_type_.request();
    const auto offset = offset_.request();
    const auto masses = masses_.request();
    const auto center_of_mass = center_of_mass_.request();
    const auto inertias = inertias_.request();
    const auto q = q_.request();
    validate_inputs(a, alpha, d, theta, joint_type, offset, masses, center_of_mass, inertias, q);
    const Transform base_transform = read_transform(base);
    const py::ssize_t samples = q.shape[0];
    const py::ssize_t joints = a.shape[0];
    auto output = output_array(requested_output, {samples, joints, joints}, "output");
    const auto* gravity = static_cast<const double*>(nullptr);
    auto* output_data = static_cast<double*>(output.request().ptr);
    {
        py::gil_scoped_release release;
        parallel_rows(samples, joints * joints, [&](py::ssize_t begin, py::ssize_t end) {
            for (py::ssize_t sample = begin; sample < end; ++sample) {
                mass_and_gravity_single(
                    static_cast<const double*>(a.ptr), static_cast<const double*>(alpha.ptr),
                    static_cast<const double*>(d.ptr), static_cast<const double*>(theta.ptr),
                    static_cast<const std::int8_t*>(joint_type.ptr), static_cast<const double*>(offset.ptr),
                    static_cast<const double*>(masses.ptr), static_cast<const double*>(center_of_mass.ptr),
                    static_cast<const double*>(inertias.ptr), static_cast<const double*>(q.ptr) + sample * joints,
                    joints, base_transform, gravity, output_data + sample * joints * joints, nullptr);
            }
        });
    }
    return output;
}

static py::array_t<double> gravity_forces_batch_dh(
    Vector a_, Vector alpha_, Vector d_, Vector theta_, JointTypes joint_type_, Vector offset_,
    Vector masses_, Batch center_of_mass_, Batch inertias_, Batch q_, Vector gravity_, py::object base,
    py::object requested_output) {
    const auto a = a_.request();
    const auto alpha = alpha_.request();
    const auto d = d_.request();
    const auto theta = theta_.request();
    const auto joint_type = joint_type_.request();
    const auto offset = offset_.request();
    const auto masses = masses_.request();
    const auto center_of_mass = center_of_mass_.request();
    const auto inertias = inertias_.request();
    const auto q = q_.request();
    const auto gravity = gravity_.request();
    validate_inputs(a, alpha, d, theta, joint_type, offset, masses, center_of_mass, inertias, q);
    if (gravity.ndim != 1 || gravity.shape[0] != 3) {
        throw std::invalid_argument("gravity must have shape (3,)");
    }
    require_finite(static_cast<const double*>(gravity.ptr), 3, "gravity");
    const Transform base_transform = read_transform(base);
    const py::ssize_t samples = q.shape[0];
    const py::ssize_t joints = a.shape[0];
    auto output = output_array(requested_output, {samples, joints}, "output");
    auto* output_data = static_cast<double*>(output.request().ptr);
    {
        py::gil_scoped_release release;
        parallel_rows(samples, joints * joints, [&](py::ssize_t begin, py::ssize_t end) {
            for (py::ssize_t sample = begin; sample < end; ++sample) {
                mass_and_gravity_single(
                    static_cast<const double*>(a.ptr), static_cast<const double*>(alpha.ptr),
                    static_cast<const double*>(d.ptr), static_cast<const double*>(theta.ptr),
                    static_cast<const std::int8_t*>(joint_type.ptr), static_cast<const double*>(offset.ptr),
                    static_cast<const double*>(masses.ptr), static_cast<const double*>(center_of_mass.ptr),
                    static_cast<const double*>(inertias.ptr), static_cast<const double*>(q.ptr) + sample * joints,
                    joints, base_transform, static_cast<const double*>(gravity.ptr), nullptr,
                    output_data + sample * joints);
            }
        });
    }
    return output;
}

static py::array_t<double> inverse_dynamics_batch_dh(
    Vector a_, Vector alpha_, Vector d_, Vector theta_, JointTypes joint_type_, Vector offset_,
    Vector masses_, Batch center_of_mass_, Batch inertias_, Batch q_, Batch qdd_, Vector gravity_,
    py::object base, py::object requested_output) {
    const auto a = a_.request();
    const auto alpha = alpha_.request();
    const auto d = d_.request();
    const auto theta = theta_.request();
    const auto joint_type = joint_type_.request();
    const auto offset = offset_.request();
    const auto masses = masses_.request();
    const auto center_of_mass = center_of_mass_.request();
    const auto inertias = inertias_.request();
    const auto q = q_.request();
    const auto qdd = qdd_.request();
    const auto gravity = gravity_.request();
    validate_inputs(a, alpha, d, theta, joint_type, offset, masses, center_of_mass, inertias, q);
    if (qdd.ndim != 2 || qdd.shape[0] != q.shape[0] || qdd.shape[1] != q.shape[1]) {
        throw std::invalid_argument("qdd must have the same shape as q");
    }
    if (gravity.ndim != 1 || gravity.shape[0] != 3) {
        throw std::invalid_argument("gravity must have shape (3,)");
    }
    require_finite(static_cast<const double*>(qdd.ptr), qdd.size, "qdd");
    require_finite(static_cast<const double*>(gravity.ptr), 3, "gravity");
    const Transform base_transform = read_transform(base);
    const py::ssize_t samples = q.shape[0];
    const py::ssize_t joints = a.shape[0];
    auto output = output_array(requested_output, {samples, joints}, "output");
    auto* output_data = static_cast<double*>(output.request().ptr);
    {
        py::gil_scoped_release release;
        parallel_rows(samples, joints * joints, [&](py::ssize_t begin, py::ssize_t end) {
            std::vector<double> matrix(static_cast<std::size_t>(joints * joints));
            std::vector<double> forces(static_cast<std::size_t>(joints));
            for (py::ssize_t sample = begin; sample < end; ++sample) {
                mass_and_gravity_single(
                    static_cast<const double*>(a.ptr), static_cast<const double*>(alpha.ptr),
                    static_cast<const double*>(d.ptr), static_cast<const double*>(theta.ptr),
                    static_cast<const std::int8_t*>(joint_type.ptr), static_cast<const double*>(offset.ptr),
                    static_cast<const double*>(masses.ptr), static_cast<const double*>(center_of_mass.ptr),
                    static_cast<const double*>(inertias.ptr), static_cast<const double*>(q.ptr) + sample * joints,
                    joints, base_transform, static_cast<const double*>(gravity.ptr), matrix.data(), forces.data());
                const double* accelerations = static_cast<const double*>(qdd.ptr) + sample * joints;
                for (py::ssize_t row = 0; row < joints; ++row) {
                    double torque = forces[row];
                    for (py::ssize_t column = 0; column < joints; ++column) {
                        torque += matrix[row * joints + column] * accelerations[column];
                    }
                    output_data[sample * joints + row] = torque;
                }
            }
        });
    }
    return output;
}

static py::tuple mass_and_gravity_batch_dh(
    Vector a_, Vector alpha_, Vector d_, Vector theta_, JointTypes joint_type_, Vector offset_,
    Vector masses_, Batch center_of_mass_, Batch inertias_, Batch q_, Vector gravity_, py::object base,
    py::object requested_matrix, py::object requested_forces) {
    const auto a = a_.request();
    const auto alpha = alpha_.request();
    const auto d = d_.request();
    const auto theta = theta_.request();
    const auto joint_type = joint_type_.request();
    const auto offset = offset_.request();
    const auto masses = masses_.request();
    const auto center_of_mass = center_of_mass_.request();
    const auto inertias = inertias_.request();
    const auto q = q_.request();
    const auto gravity = gravity_.request();
    validate_inputs(a, alpha, d, theta, joint_type, offset, masses, center_of_mass, inertias, q);
    if (gravity.ndim != 1 || gravity.shape[0] != 3) {
        throw std::invalid_argument("gravity must have shape (3,)");
    }
    require_finite(static_cast<const double*>(gravity.ptr), 3, "gravity");
    const Transform base_transform = read_transform(base);
    const py::ssize_t samples = q.shape[0];
    const py::ssize_t joints = a.shape[0];
    auto matrices = output_array(requested_matrix, {samples, joints, joints}, "matrix_output");
    auto forces = output_array(requested_forces, {samples, joints}, "force_output");
    auto* matrix_data = static_cast<double*>(matrices.request().ptr);
    auto* force_data = static_cast<double*>(forces.request().ptr);
    {
        py::gil_scoped_release release;
        parallel_rows(samples, joints * joints, [&](py::ssize_t begin, py::ssize_t end) {
            for (py::ssize_t sample = begin; sample < end; ++sample) {
                mass_and_gravity_single(
                    static_cast<const double*>(a.ptr), static_cast<const double*>(alpha.ptr),
                    static_cast<const double*>(d.ptr), static_cast<const double*>(theta.ptr),
                    static_cast<const std::int8_t*>(joint_type.ptr), static_cast<const double*>(offset.ptr),
                    static_cast<const double*>(masses.ptr), static_cast<const double*>(center_of_mass.ptr),
                    static_cast<const double*>(inertias.ptr), static_cast<const double*>(q.ptr) + sample * joints,
                    joints, base_transform, static_cast<const double*>(gravity.ptr),
                    matrix_data + sample * joints * joints, force_data + sample * joints);
            }
        });
    }
    return py::make_tuple(matrices, forces);
}

PYBIND11_MODULE(_dynamics_cpp, m) {
    m.doc() = "C++ accelerated batched rigid-body dynamics for serial DH chains";
    m.def(
        "mass_matrix_batch_dh", &mass_matrix_batch_dh,
        py::arg("a"), py::arg("alpha"), py::arg("d"), py::arg("theta"), py::arg("joint_type"),
        py::arg("offset"), py::arg("masses"), py::arg("center_of_mass"), py::arg("inertias"),
        py::arg("q"), py::arg("base") = py::none(), py::arg("output") = py::none());
    m.def(
        "gravity_forces_batch_dh", &gravity_forces_batch_dh,
        py::arg("a"), py::arg("alpha"), py::arg("d"), py::arg("theta"), py::arg("joint_type"),
        py::arg("offset"), py::arg("masses"), py::arg("center_of_mass"), py::arg("inertias"),
        py::arg("q"), py::arg("gravity"), py::arg("base") = py::none(), py::arg("output") = py::none());
    m.def(
        "inverse_dynamics_batch_dh", &inverse_dynamics_batch_dh,
        py::arg("a"), py::arg("alpha"), py::arg("d"), py::arg("theta"), py::arg("joint_type"),
        py::arg("offset"), py::arg("masses"), py::arg("center_of_mass"), py::arg("inertias"),
        py::arg("q"), py::arg("qdd"), py::arg("gravity"), py::arg("base") = py::none(),
        py::arg("output") = py::none());
    m.def(
        "mass_and_gravity_batch_dh", &mass_and_gravity_batch_dh,
        py::arg("a"), py::arg("alpha"), py::arg("d"), py::arg("theta"), py::arg("joint_type"),
        py::arg("offset"), py::arg("masses"), py::arg("center_of_mass"), py::arg("inertias"),
        py::arg("q"), py::arg("gravity"), py::arg("base") = py::none(),
        py::arg("matrix_output") = py::none(), py::arg("force_output") = py::none());
}
