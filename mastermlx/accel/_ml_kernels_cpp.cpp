#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace py = pybind11;

using Matrix = py::array_t<double, py::array::c_style | py::array::forcecast>;
using Vector = py::array_t<double, py::array::c_style | py::array::forcecast>;
using Labels = py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>;

static py::buffer_info require_matrix(const Matrix& value, const char* name) {
    auto info = value.request();
    if (info.ndim != 2) {
        throw std::invalid_argument(std::string(name) + " must be a 2D array");
    }
    if (info.shape[0] <= 0 || info.shape[1] <= 0) {
        throw std::invalid_argument(std::string(name) + " must be non-empty");
    }
    const auto* data = static_cast<const double*>(info.ptr);
    const py::ssize_t size = info.shape[0] * info.shape[1];
    for (py::ssize_t i = 0; i < size; ++i) {
        if (!std::isfinite(data[i])) {
            throw std::invalid_argument(std::string(name) + " must contain only finite values");
        }
    }
    return info;
}

static py::buffer_info require_imputation_matrix(
    const Matrix& value, const char* name) {
    auto info = value.request();
    if (info.ndim != 2) {
        throw std::invalid_argument(std::string(name) + " must be a 2D array");
    }
    if (info.shape[0] <= 0 || info.shape[1] <= 0) {
        throw std::invalid_argument(std::string(name) + " must be non-empty");
    }
    const auto* data = static_cast<const double*>(info.ptr);
    const py::ssize_t size = info.shape[0] * info.shape[1];
    for (py::ssize_t i = 0; i < size; ++i) {
        if (std::isinf(data[i])) {
            throw std::invalid_argument(std::string(name) + " must not contain infinite values");
        }
    }
    return info;
}

static py::buffer_info require_vector(const Vector& value, const char* name, py::ssize_t expected) {
    auto info = value.request();
    if (info.ndim != 1 || info.shape[0] != expected) {
        throw std::invalid_argument(std::string(name) + " must match the expected shape");
    }
    const auto* data = static_cast<const double*>(info.ptr);
    for (py::ssize_t i = 0; i < expected; ++i) {
        if (!std::isfinite(data[i])) {
            throw std::invalid_argument(std::string(name) + " must contain only finite values");
        }
    }
    return info;
}

static void require_same_features(const py::buffer_info& Xb, const py::buffer_info& Yb) {
    if (Xb.shape[1] != Yb.shape[1]) {
        throw std::invalid_argument("arrays must have the same number of features");
    }
}

static void require_hmm_inputs(
    const py::buffer_info& sequence,
    const py::buffer_info& start,
    const py::buffer_info& trans,
    const py::buffer_info& emit) {
    if (sequence.ndim != 1 || sequence.shape[0] <= 0
        || start.ndim != 1 || trans.ndim != 2 || emit.ndim != 2
        || trans.shape[0] != trans.shape[1]
        || start.shape[0] != trans.shape[0]
        || emit.shape[0] != trans.shape[0]) {
        throw std::invalid_argument("invalid HMM array shapes");
    }
    const auto* start_values = static_cast<const double*>(start.ptr);
    const auto* transition = static_cast<const double*>(trans.ptr);
    const auto* emission = static_cast<const double*>(emit.ptr);
    for (py::ssize_t i = 0; i < start.shape[0]; ++i) {
        if (start_values[i] < 0.0) {
            throw std::invalid_argument("HMM probabilities must be non-negative");
        }
    }
    for (py::ssize_t i = 0; i < trans.shape[0] * trans.shape[1]; ++i) {
        if (transition[i] < 0.0) {
            throw std::invalid_argument("HMM probabilities must be non-negative");
        }
    }
    for (py::ssize_t i = 0; i < emit.shape[0] * emit.shape[1]; ++i) {
        if (emission[i] < 0.0) {
            throw std::invalid_argument("HMM probabilities must be non-negative");
        }
    }
    for (py::ssize_t i = 0; i < sequence.shape[0]; ++i) {
        const auto observation = static_cast<const std::int64_t*>(sequence.ptr)[i];
        if (observation < 0 || observation >= emit.shape[1]) {
            throw std::invalid_argument("observation index out of range");
        }
    }
}

static double log_sum_exp(const double* values, py::ssize_t size) {
    double maximum = -std::numeric_limits<double>::infinity();
    for (py::ssize_t i = 0; i < size; ++i) {
        maximum = std::max(maximum, values[i]);
    }
    if (!std::isfinite(maximum)) {
        return maximum;
    }
    double total = 0.0;
    for (py::ssize_t i = 0; i < size; ++i) {
        total += std::exp(values[i] - maximum);
    }
    return maximum + std::log(total);
}

py::array_t<double> hmm_forward(Labels sequence_, Vector start_, Matrix trans_, Matrix emit_) {
    const auto sequence = sequence_.request();
    const auto start = start_.request();
    const auto trans = trans_.request();
    const auto emit = emit_.request();
    require_hmm_inputs(sequence, start, trans, emit);
    const auto* observations = static_cast<const std::int64_t*>(sequence.ptr);
    const auto* start_values = static_cast<const double*>(start.ptr);
    const auto* transition = static_cast<const double*>(trans.ptr);
    const auto* emission = static_cast<const double*>(emit.ptr);
    const py::ssize_t time = sequence.shape[0];
    const py::ssize_t states = trans.shape[0];
    py::array_t<double> output({time, states});
    auto* result = static_cast<double*>(output.request().ptr);
    {
        py::gil_scoped_release release;
        for (py::ssize_t state = 0; state < states; ++state) {
            result[state] = std::log(start_values[state] + 1e-12)
                + std::log(emission[state * emit.shape[1] + observations[0]] + 1e-12);
        }
        std::vector<double> scratch(static_cast<std::size_t>(states));
        for (py::ssize_t step = 1; step < time; ++step) {
            const auto observation = observations[step];
            for (py::ssize_t destination = 0; destination < states; ++destination) {
                for (py::ssize_t source = 0; source < states; ++source) {
                    scratch[static_cast<std::size_t>(source)] =
                        result[(step - 1) * states + source]
                        + std::log(transition[source * states + destination] + 1e-12);
                }
                result[step * states + destination] =
                    std::log(emission[destination * emit.shape[1] + observation] + 1e-12)
                    + log_sum_exp(scratch.data(), states);
            }
        }
    }
    return output;
}

py::array_t<double> hmm_backward(Labels sequence_, Vector start_, Matrix trans_, Matrix emit_) {
    const auto sequence = sequence_.request();
    const auto start = start_.request();
    const auto trans = trans_.request();
    const auto emit = emit_.request();
    require_hmm_inputs(sequence, start, trans, emit);
    const auto* observations = static_cast<const std::int64_t*>(sequence.ptr);
    const auto* transition = static_cast<const double*>(trans.ptr);
    const auto* emission = static_cast<const double*>(emit.ptr);
    const py::ssize_t time = sequence.shape[0];
    const py::ssize_t states = trans.shape[0];
    py::array_t<double> output({time, states});
    auto* result = static_cast<double*>(output.request().ptr);
    {
        py::gil_scoped_release release;
        for (py::ssize_t state = 0; state < states; ++state) {
            result[(time - 1) * states + state] = 0.0;
        }
        std::vector<double> scratch(static_cast<std::size_t>(states));
        for (py::ssize_t step = time - 2; step >= 0; --step) {
            const auto observation = observations[step + 1];
            for (py::ssize_t source = 0; source < states; ++source) {
                for (py::ssize_t destination = 0; destination < states; ++destination) {
                    scratch[static_cast<std::size_t>(destination)] =
                        std::log(transition[source * states + destination] + 1e-12)
                        + std::log(emission[destination * emit.shape[1] + observation] + 1e-12)
                        + result[(step + 1) * states + destination];
                }
                result[step * states + source] = log_sum_exp(scratch.data(), states);
            }
        }
    }
    return output;
}

py::array_t<std::int64_t> hmm_viterbi(
    Labels sequence_, Vector start_, Matrix trans_, Matrix emit_) {
    const auto sequence = sequence_.request();
    const auto start = start_.request();
    const auto trans = trans_.request();
    const auto emit = emit_.request();
    require_hmm_inputs(sequence, start, trans, emit);
    const auto* observations = static_cast<const std::int64_t*>(sequence.ptr);
    const auto* start_values = static_cast<const double*>(start.ptr);
    const auto* transition = static_cast<const double*>(trans.ptr);
    const auto* emission = static_cast<const double*>(emit.ptr);
    const py::ssize_t time = sequence.shape[0];
    const py::ssize_t states = trans.shape[0];
    py::array_t<std::int64_t> path(time);
    py::array_t<double> delta({time, states});
    py::array_t<std::int64_t> psi({time, states});
    auto* output = static_cast<std::int64_t*>(path.request().ptr);
    auto* scores = static_cast<double*>(delta.request().ptr);
    auto* backpointers = static_cast<std::int64_t*>(psi.request().ptr);
    {
        py::gil_scoped_release release;
        for (py::ssize_t state = 0; state < states; ++state) {
            scores[state] = std::log(start_values[state] + 1e-12)
                + std::log(emission[state * emit.shape[1] + observations[0]] + 1e-12);
            backpointers[state] = 0;
        }
        for (py::ssize_t step = 1; step < time; ++step) {
            const auto observation = observations[step];
            for (py::ssize_t destination = 0; destination < states; ++destination) {
                double best = -std::numeric_limits<double>::infinity();
                std::int64_t best_state = 0;
                for (py::ssize_t source = 0; source < states; ++source) {
                    const double candidate = scores[(step - 1) * states + source]
                        + std::log(transition[source * states + destination] + 1e-12);
                    if (candidate > best) {
                        best = candidate;
                        best_state = source;
                    }
                }
                scores[step * states + destination] = best
                    + std::log(emission[destination * emit.shape[1] + observation] + 1e-12);
                backpointers[step * states + destination] = best_state;
            }
        }
        std::int64_t last = 0;
        for (py::ssize_t state = 1; state < states; ++state) {
            if (scores[(time - 1) * states + state] > scores[(time - 1) * states + last]) {
                last = state;
            }
        }
        output[time - 1] = last;
        for (py::ssize_t step = time - 2; step >= 0; --step) {
            last = backpointers[(step + 1) * states + last];
            output[step] = last;
        }
    }
    return path;
}

static bool neighbor_distance_less(
    const std::pair<double, py::ssize_t>& left,
    const std::pair<double, py::ssize_t>& right) {
    if (left.first != right.first) {
        return left.first < right.first;
    }
    return left.second < right.second;
}

static void select_nearest_neighbors(
    std::vector<std::pair<double, py::ssize_t>>& distances,
    py::ssize_t n_neighbors) {
    const auto k = static_cast<std::size_t>(n_neighbors);
    if (k < distances.size()) {
        std::nth_element(
            distances.begin(),
            distances.begin() + static_cast<std::ptrdiff_t>(k),
            distances.end(),
            neighbor_distance_less);
    }
    std::sort(
        distances.begin(),
        distances.begin() + static_cast<std::ptrdiff_t>(k),
        neighbor_distance_less);
}

template <typename Fn>
static void parallel_rows(py::ssize_t rows, py::ssize_t work, Fn&& fn) {
    if (rows < 4 || work < 1 || static_cast<long double>(rows) * work < 1048576.0L) {
        fn(0, rows);
        return;
    }
    const unsigned hardware = std::thread::hardware_concurrency();
    const unsigned available = hardware == 0 ? 1u : hardware;
    const unsigned workers = std::min<unsigned>(std::min<unsigned>(available, 4u), static_cast<unsigned>(rows));
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

// ---------------------------------------------------------------------------
// Semi-supervised graph construction.
// ---------------------------------------------------------------------------

py::array_t<double> rbf_affinity(Matrix X_, double gamma) {
    if (!std::isfinite(gamma)) {
        throw std::invalid_argument("gamma must be finite");
    }
    const auto Xb = require_matrix(X_, "X");
    const auto* X = static_cast<const double*>(Xb.ptr);
    const py::ssize_t n = Xb.shape[0];
    const py::ssize_t d = Xb.shape[1];
    py::array_t<double> out({n, n});
    auto* result = static_cast<double*>(out.request().ptr);
    {
        py::gil_scoped_release release;
        parallel_rows(n, n, [&](py::ssize_t begin, py::ssize_t end) {
            for (py::ssize_t i = begin; i < end; ++i) {
                const double* xi = X + i * d;
                double* row = result + i * n;
                for (py::ssize_t j = 0; j < n; ++j) {
                    const double* xj = X + j * d;
                    double d2 = 0.0;
                    for (py::ssize_t k = 0; k < d; ++k) {
                        const double diff = xi[k] - xj[k];
                        d2 += diff * diff;
                    }
                    row[j] = (i == j) ? 0.0 : std::exp(-gamma * d2);
                }
            }
        });
    }
    return out;
}

py::array_t<double> knn_affinity(Matrix X_, py::ssize_t n_neighbors) {
    const auto Xb = require_matrix(X_, "X");
    const py::ssize_t n = Xb.shape[0];
    const py::ssize_t d = Xb.shape[1];
    if (n_neighbors < 1 || n_neighbors >= n) {
        throw std::invalid_argument("n_neighbors must be between 1 and n_samples - 1");
    }
    const auto* X = static_cast<const double*>(Xb.ptr);
    py::array_t<double> out({n, n});
    auto* result = static_cast<double*>(out.request().ptr);
    std::fill(result, result + n * n, 0.0);
    {
        py::gil_scoped_release release;
        parallel_rows(n, n, [&](py::ssize_t begin, py::ssize_t end) {
            std::vector<std::pair<double, py::ssize_t>> distances;
            distances.reserve(static_cast<std::size_t>(n - 1));
            for (py::ssize_t i = begin; i < end; ++i) {
                distances.clear();
                const double* xi = X + i * d;
                for (py::ssize_t j = 0; j < n; ++j) {
                    if (i == j) {
                        continue;
                    }
                    const double* xj = X + j * d;
                    double d2 = 0.0;
                    for (py::ssize_t k = 0; k < d; ++k) {
                        const double diff = xi[k] - xj[k];
                        d2 += diff * diff;
                    }
                    distances.emplace_back(d2, j);
                }
                select_nearest_neighbors(distances, n_neighbors);
                double* row = result + i * n;
                for (py::ssize_t p = 0; p < n_neighbors; ++p) {
                    row[distances[static_cast<std::size_t>(p)].second] = 1.0;
                }
            }
        });
        // Match the existing public behavior: a mutual edge is not required.
        // Symmetrization is performed after all rows have been populated.
        for (py::ssize_t i = 0; i < n; ++i) {
            for (py::ssize_t j = i + 1; j < n; ++j) {
                const double edge = std::max(result[i * n + j], result[j * n + i]);
                result[i * n + j] = edge;
                result[j * n + i] = edge;
            }
        }
    }
    return out;
}

static std::vector<std::vector<py::ssize_t>> build_knn_adjacency(
    const double* X, py::ssize_t n, py::ssize_t d, py::ssize_t n_neighbors) {
    std::vector<std::vector<py::ssize_t>> directed(static_cast<std::size_t>(n));
    parallel_rows(n, n, [&](py::ssize_t begin, py::ssize_t end) {
        std::vector<std::pair<double, py::ssize_t>> distances;
        distances.reserve(static_cast<std::size_t>(n - 1));
        for (py::ssize_t i = begin; i < end; ++i) {
            distances.clear();
            const double* xi = X + i * d;
            for (py::ssize_t j = 0; j < n; ++j) {
                if (i == j) {
                    continue;
                }
                const double* xj = X + j * d;
                double d2 = 0.0;
                for (py::ssize_t q = 0; q < d; ++q) {
                    const double diff = xi[q] - xj[q];
                    d2 += diff * diff;
                }
                distances.emplace_back(d2, j);
            }
            select_nearest_neighbors(distances, n_neighbors);
            auto& row = directed[static_cast<std::size_t>(i)];
            row.reserve(static_cast<std::size_t>(n_neighbors));
            for (py::ssize_t p = 0; p < n_neighbors; ++p) {
                row.push_back(distances[static_cast<std::size_t>(p)].second);
            }
        }
    });
    for (py::ssize_t i = 0; i < n; ++i) {
        for (const py::ssize_t neighbor : directed[static_cast<std::size_t>(i)]) {
            directed[static_cast<std::size_t>(neighbor)].push_back(i);
        }
    }
    for (auto& row : directed) {
        std::sort(row.begin(), row.end());
        row.erase(std::unique(row.begin(), row.end()), row.end());
    }
    return directed;
}

py::tuple knn_graph(Matrix X_, py::ssize_t n_neighbors) {
    const auto Xb = require_matrix(X_, "X");
    const py::ssize_t n = Xb.shape[0];
    const py::ssize_t d = Xb.shape[1];
    if (n_neighbors < 1 || n_neighbors >= n) {
        throw std::invalid_argument("n_neighbors must be between 1 and n_samples - 1");
    }
    const auto* X = static_cast<const double*>(Xb.ptr);
    const auto adjacency = build_knn_adjacency(X, n, d, n_neighbors);
    py::array_t<std::int64_t> indptr(n + 1);
    auto* rows = static_cast<std::int64_t*>(indptr.request().ptr);
    rows[0] = 0;
    for (py::ssize_t i = 0; i < n; ++i) {
        rows[i + 1] = rows[i] + static_cast<std::int64_t>(adjacency[static_cast<std::size_t>(i)].size());
    }
    const py::ssize_t edges = static_cast<py::ssize_t>(rows[n]);
    py::array_t<std::int64_t> indices(edges);
    py::array_t<double> values(edges);
    auto* cols = static_cast<std::int64_t*>(indices.request().ptr);
    auto* data = static_cast<double*>(values.request().ptr);
    py::ssize_t offset = 0;
    for (const auto& row : adjacency) {
        for (const py::ssize_t neighbor : row) {
            cols[offset] = static_cast<std::int64_t>(neighbor);
            data[offset] = 1.0;
            ++offset;
        }
    }
    return py::make_tuple(indptr, indices, values);
}

py::array_t<double> knn_impute(
    Matrix X_, Matrix X_fit_, py::ssize_t n_neighbors, bool distance_weighted) {
    const auto Xb = require_imputation_matrix(X_, "X");
    const auto Fb = require_imputation_matrix(X_fit_, "X_fit");
    if (Xb.shape[1] != Fb.shape[1]) {
        throw std::invalid_argument("X and X_fit must have the same number of features");
    }
    if (n_neighbors < 1) {
        throw std::invalid_argument("n_neighbors must be at least 1");
    }

    const auto* X = static_cast<const double*>(Xb.ptr);
    const auto* X_fit = static_cast<const double*>(Fb.ptr);
    const py::ssize_t n_query = Xb.shape[0];
    const py::ssize_t n_train = Fb.shape[0];
    const py::ssize_t d = Xb.shape[1];
    py::array_t<double> out({n_query, d});
    auto* result = static_cast<double*>(out.request().ptr);
    std::copy(X, X + n_query * d, result);

    {
        py::gil_scoped_release release;
        parallel_rows(n_query, d, [&](py::ssize_t begin, py::ssize_t end) {
            std::vector<std::pair<double, py::ssize_t>> candidates;
            candidates.reserve(static_cast<std::size_t>(n_train));
            for (py::ssize_t row = begin; row < end; ++row) {
                const double* query = X + row * d;
                double* target = result + row * d;
                for (py::ssize_t column = 0; column < d; ++column) {
                    if (!std::isnan(query[column])) {
                        continue;
                    }
                    candidates.clear();
                    for (py::ssize_t train_row = 0; train_row < n_train; ++train_row) {
                        const double value = X_fit[train_row * d + column];
                        if (!std::isfinite(value)) {
                            continue;
                        }
                        const double* train = X_fit + train_row * d;
                        double distance = 0.0;
                        py::ssize_t overlap = 0;
                        for (py::ssize_t feature = 0; feature < d; ++feature) {
                            if (!std::isfinite(query[feature])
                                || !std::isfinite(train[feature])) {
                                continue;
                            }
                            const double diff = query[feature] - train[feature];
                            distance += diff * diff;
                            ++overlap;
                        }
                        if (overlap > 0) {
                            candidates.emplace_back(distance, train_row);
                        }
                    }
                    if (candidates.empty()) {
                        continue;
                    }
                    const py::ssize_t k = std::min(
                        n_neighbors, static_cast<py::ssize_t>(candidates.size()));
                    select_nearest_neighbors(candidates, k);
                    if (!distance_weighted) {
                        double total = 0.0;
                        for (py::ssize_t index = 0; index < k; ++index) {
                            total += X_fit[candidates[static_cast<std::size_t>(index)].second * d + column];
                        }
                        target[column] = total / static_cast<double>(k);
                    } else {
                        double weighted_total = 0.0;
                        double weight_total = 0.0;
                        for (py::ssize_t index = 0; index < k; ++index) {
                            const auto& candidate = candidates[static_cast<std::size_t>(index)];
                            const double weight = 1.0 / std::max(std::sqrt(candidate.first), 1e-12);
                            weighted_total += weight * X_fit[candidate.second * d + column];
                            weight_total += weight;
                        }
                        target[column] = weighted_total / weight_total;
                    }
                }
            }
        });
    }
    return out;
}

py::tuple radius_neighbors(Matrix X_, Matrix X_fit_, double radius) {
    if (!std::isfinite(radius) || radius <= 0.0) {
        throw std::invalid_argument("radius must be positive and finite");
    }
    const auto Xb = require_matrix(X_, "X");
    const auto Fb = require_matrix(X_fit_, "X_fit");
    if (Xb.shape[1] != Fb.shape[1]) {
        throw std::invalid_argument("X and X_fit must have the same number of features");
    }
    const auto* X = static_cast<const double*>(Xb.ptr);
    const auto* X_fit = static_cast<const double*>(Fb.ptr);
    const py::ssize_t n_query = Xb.shape[0];
    const py::ssize_t n_train = Fb.shape[0];
    const py::ssize_t d = Xb.shape[1];
    const double radius_sq = radius * radius;
    std::vector<std::vector<std::pair<py::ssize_t, double>>> adjacency(
        static_cast<std::size_t>(n_query));
    {
        py::gil_scoped_release release;
        parallel_rows(n_query, n_train, [&](py::ssize_t begin, py::ssize_t end) {
            for (py::ssize_t row = begin; row < end; ++row) {
                const double* query = X + row * d;
                auto& neighbors = adjacency[static_cast<std::size_t>(row)];
                for (py::ssize_t train_row = 0; train_row < n_train; ++train_row) {
                    const double* train = X_fit + train_row * d;
                    double distance_sq = 0.0;
                    for (py::ssize_t feature = 0; feature < d; ++feature) {
                        const double diff = query[feature] - train[feature];
                        distance_sq += diff * diff;
                    }
                    if (distance_sq <= radius_sq) {
                        neighbors.emplace_back(train_row, std::sqrt(std::max(distance_sq, 0.0)));
                    }
                }
            }
        });
    }

    py::array_t<std::int64_t> indptr(n_query + 1);
    auto* rows = static_cast<std::int64_t*>(indptr.request().ptr);
    rows[0] = 0;
    for (py::ssize_t row = 0; row < n_query; ++row) {
        rows[row + 1] = rows[row] + static_cast<std::int64_t>(
            adjacency[static_cast<std::size_t>(row)].size());
    }
    const py::ssize_t edges = static_cast<py::ssize_t>(rows[n_query]);
    py::array_t<std::int64_t> indices(edges);
    py::array_t<double> values(edges);
    auto* cols = static_cast<std::int64_t*>(indices.request().ptr);
    auto* data = static_cast<double*>(values.request().ptr);
    py::ssize_t offset = 0;
    for (const auto& row : adjacency) {
        for (const auto& neighbor : row) {
            cols[offset] = static_cast<std::int64_t>(neighbor.first);
            data[offset] = neighbor.second;
            ++offset;
        }
    }
    return py::make_tuple(indptr, indices, values);
}

py::tuple dbscan_neighbors(Matrix X_, double eps) {
    if (!std::isfinite(eps) || eps <= 0.0) {
        throw std::invalid_argument("eps must be positive and finite");
    }
    const auto Xb = require_matrix(X_, "X");
    const auto* X = static_cast<const double*>(Xb.ptr);
    const py::ssize_t n = Xb.shape[0];
    const py::ssize_t d = Xb.shape[1];
    const double radius_sq = eps * eps;
    std::vector<std::vector<py::ssize_t>> adjacency(static_cast<std::size_t>(n));
    {
        py::gil_scoped_release release;
        parallel_rows(n, n, [&](py::ssize_t begin, py::ssize_t end) {
            for (py::ssize_t i = begin; i < end; ++i) {
                const double* xi = X + i * d;
                auto& row = adjacency[static_cast<std::size_t>(i)];
                for (py::ssize_t j = 0; j < n; ++j) {
                    const double* xj = X + j * d;
                    double d2 = 0.0;
                    for (py::ssize_t q = 0; q < d; ++q) {
                        const double diff = xi[q] - xj[q];
                        d2 += diff * diff;
                    }
                    if (d2 <= radius_sq) {
                        row.push_back(j);
                    }
                }
            }
        });
    }
    py::array_t<std::int64_t> indptr(n + 1);
    auto* rows = static_cast<std::int64_t*>(indptr.request().ptr);
    rows[0] = 0;
    for (py::ssize_t i = 0; i < n; ++i) {
        rows[i + 1] = rows[i] + static_cast<std::int64_t>(adjacency[static_cast<std::size_t>(i)].size());
    }
    const py::ssize_t edges = static_cast<py::ssize_t>(rows[n]);
    py::array_t<std::int64_t> indices(edges);
    py::array_t<double> values(edges);
    auto* cols = static_cast<std::int64_t*>(indices.request().ptr);
    auto* data = static_cast<double*>(values.request().ptr);
    py::ssize_t offset = 0;
    for (const auto& row : adjacency) {
        for (const py::ssize_t neighbor : row) {
            cols[offset] = static_cast<std::int64_t>(neighbor);
            data[offset] = 1.0;
            ++offset;
        }
    }
    return py::make_tuple(indptr, indices, values);
}

py::tuple dbscan_labels(
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> indptr_,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> indices_,
    py::ssize_t min_samples) {
    const auto indptr = indptr_.request();
    const auto indices = indices_.request();
    if (indptr.ndim != 1 || indices.ndim != 1 || indptr.shape[0] < 2
        || min_samples < 1) {
        throw std::invalid_argument("invalid CSR arrays or min_samples");
    }
    const py::ssize_t n = indptr.shape[0] - 1;
    const py::ssize_t edges = indices.shape[0];
    const auto* rows = static_cast<const std::int64_t*>(indptr.ptr);
    const auto* cols = static_cast<const std::int64_t*>(indices.ptr);
    if (rows[0] != 0 || rows[n] != edges) {
        throw std::invalid_argument("invalid CSR row pointer");
    }
    for (py::ssize_t row = 0; row < n; ++row) {
        if (rows[row] < 0 || rows[row + 1] < rows[row]) {
            throw std::invalid_argument("indptr must be non-decreasing and non-negative");
        }
    }
    for (py::ssize_t edge = 0; edge < edges; ++edge) {
        if (cols[edge] < 0 || cols[edge] >= n) {
            throw std::invalid_argument("invalid CSR indices");
        }
    }

    py::array_t<std::int64_t> labels(n);
    py::array_t<std::int64_t> core_samples;
    auto* output = static_cast<std::int64_t*>(labels.request().ptr);
    std::fill(output, output + n, static_cast<std::int64_t>(-1));
    std::vector<char> core(static_cast<std::size_t>(n), 0);
    std::vector<char> visited(static_cast<std::size_t>(n), 0);
    std::vector<py::ssize_t> core_indices;
    core_indices.reserve(static_cast<std::size_t>(n));
    for (py::ssize_t row = 0; row < n; ++row) {
        if (rows[row + 1] - rows[row] >= min_samples) {
            core[static_cast<std::size_t>(row)] = 1;
            core_indices.push_back(row);
        }
    }
    core_samples = py::array_t<std::int64_t>(static_cast<py::ssize_t>(core_indices.size()));
    auto* core_output = static_cast<std::int64_t*>(core_samples.request().ptr);
    for (std::size_t i = 0; i < core_indices.size(); ++i) {
        core_output[i] = static_cast<std::int64_t>(core_indices[i]);
    }

    std::int64_t cluster_id = 0;
    std::vector<py::ssize_t> stack;
    stack.reserve(static_cast<std::size_t>(n));
    {
        py::gil_scoped_release release;
        for (py::ssize_t point = 0; point < n; ++point) {
            if (visited[static_cast<std::size_t>(point)]
                || !core[static_cast<std::size_t>(point)]) {
                continue;
            }
            stack.clear();
            stack.push_back(point);
            visited[static_cast<std::size_t>(point)] = 1;
            output[point] = cluster_id;
            while (!stack.empty()) {
                const py::ssize_t current = stack.back();
                stack.pop_back();
                for (std::int64_t edge = rows[current]; edge < rows[current + 1]; ++edge) {
                    const py::ssize_t neighbor = static_cast<py::ssize_t>(cols[edge]);
                    if (output[neighbor] == -1) {
                        output[neighbor] = cluster_id;
                    }
                    if (!visited[static_cast<std::size_t>(neighbor)]) {
                        visited[static_cast<std::size_t>(neighbor)] = 1;
                        if (core[static_cast<std::size_t>(neighbor)]) {
                            stack.push_back(neighbor);
                        }
                    }
                }
            }
            ++cluster_id;
        }
    }
    return py::make_tuple(labels, core_samples);
}

py::array_t<double> meanshift_update(Matrix X_, Matrix centers_, double bandwidth) {
    if (!std::isfinite(bandwidth) || bandwidth <= 0.0) {
        throw std::invalid_argument("bandwidth must be positive and finite");
    }
    const auto Xb = require_matrix(X_, "X");
    const auto Cb = require_matrix(centers_, "centers");
    if (Xb.shape[1] != Cb.shape[1]) {
        throw std::invalid_argument("X and centers must have the same number of features");
    }
    const auto* X = static_cast<const double*>(Xb.ptr);
    const auto* centers = static_cast<const double*>(Cb.ptr);
    const py::ssize_t n_samples = Xb.shape[0];
    const py::ssize_t n_centers = Cb.shape[0];
    const py::ssize_t d = Xb.shape[1];
    const double bandwidth_sq = bandwidth * bandwidth;
    py::array_t<double> out({n_centers, d});
    auto* result = static_cast<double*>(out.request().ptr);
    std::copy(centers, centers + n_centers * d, result);
    {
        py::gil_scoped_release release;
        parallel_rows(n_centers, n_samples, [&](py::ssize_t begin, py::ssize_t end) {
            std::vector<double> sums(static_cast<std::size_t>(d));
            for (py::ssize_t row = begin; row < end; ++row) {
                std::fill(sums.begin(), sums.end(), 0.0);
                const double* center = centers + row * d;
                py::ssize_t count = 0;
                for (py::ssize_t sample = 0; sample < n_samples; ++sample) {
                    const double* point = X + sample * d;
                    double distance_sq = 0.0;
                    for (py::ssize_t feature = 0; feature < d; ++feature) {
                        const double diff = point[feature] - center[feature];
                        distance_sq += diff * diff;
                    }
                    if (distance_sq <= bandwidth_sq) {
                        for (py::ssize_t feature = 0; feature < d; ++feature) {
                            sums[static_cast<std::size_t>(feature)] += point[feature];
                        }
                        ++count;
                    }
                }
                if (count > 0) {
                    double* output = result + row * d;
                    for (py::ssize_t feature = 0; feature < d; ++feature) {
                        output[feature] = sums[static_cast<std::size_t>(feature)]
                            / static_cast<double>(count);
                    }
                }
            }
        });
    }
    return out;
}

py::array_t<double> csr_propagate(
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> indptr_,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> indices_,
    Vector weights_, Matrix F_) {
    const auto ib = indptr_.request();
    const auto cb = indices_.request();
    const auto wb = weights_.request();
    const auto Fb = require_matrix(F_, "F");
    if (ib.ndim != 1 || cb.ndim != 1 || wb.ndim != 1 || ib.shape[0] < 1
        || cb.shape[0] != wb.shape[0] || ib.shape[0] != Fb.shape[0] + 1
        || ib.shape[0] < 1 || ib.shape[ib.ndim - 1] < 0) {
        throw std::invalid_argument("invalid CSR arrays");
    }
    const auto* rows = static_cast<const std::int64_t*>(ib.ptr);
    const auto* cols = static_cast<const std::int64_t*>(cb.ptr);
    const auto* weights = static_cast<const double*>(wb.ptr);
    const auto* F = static_cast<const double*>(Fb.ptr);
    const py::ssize_t n = Fb.shape[0];
    const py::ssize_t c = Fb.shape[1];
    const py::ssize_t edges = cb.shape[0];
    if (rows[0] != 0 || rows[n] != edges) {
        throw std::invalid_argument("invalid CSR row pointer");
    }
    for (py::ssize_t i = 0; i < n; ++i) {
        if (rows[i] < 0 || rows[i + 1] < rows[i]) {
            throw std::invalid_argument("indptr must be non-decreasing and non-negative");
        }
    }
    for (py::ssize_t e = 0; e < edges; ++e) {
        if (cols[e] < 0 || cols[e] >= n || !std::isfinite(weights[e])) {
            throw std::invalid_argument("invalid CSR indices or weights");
        }
    }
    py::array_t<double> out({n, c});
    auto* result = static_cast<double*>(out.request().ptr);
    std::fill(result, result + n * c, 0.0);
    {
        py::gil_scoped_release release;
        parallel_rows(n, c, [&](py::ssize_t begin, py::ssize_t end) {
            for (py::ssize_t i = begin; i < end; ++i) {
                double* target = result + i * c;
                for (std::int64_t e = rows[i]; e < rows[i + 1]; ++e) {
                    const double* source = F + cols[e] * c;
                    for (py::ssize_t q = 0; q < c; ++q) {
                        target[q] += weights[e] * source[q];
                    }
                }
            }
        });
    }
    return out;
}

// ---------------------------------------------------------------------------
// K-means assignment and reduction.
// ---------------------------------------------------------------------------

py::tuple kmeans_assign(Matrix X_, Matrix centers_) {
    const auto Xb = require_matrix(X_, "X");
    const auto Cb = require_matrix(centers_, "centers");
    require_same_features(Xb, Cb);
    const auto* X = static_cast<const double*>(Xb.ptr);
    const auto* centers = static_cast<const double*>(Cb.ptr);
    const py::ssize_t n = Xb.shape[0];
    const py::ssize_t k = Cb.shape[0];
    const py::ssize_t d = Xb.shape[1];
    py::array_t<std::int64_t> labels(n);
    py::array_t<double> distances(n);
    auto* label_data = static_cast<std::int64_t*>(labels.request().ptr);
    auto* distance_data = static_cast<double*>(distances.request().ptr);
    {
        py::gil_scoped_release release;
        parallel_rows(n, k, [&](py::ssize_t begin, py::ssize_t end) {
            for (py::ssize_t i = begin; i < end; ++i) {
                const double* xi = X + i * d;
                double best = std::numeric_limits<double>::infinity();
                py::ssize_t best_index = 0;
                for (py::ssize_t j = 0; j < k; ++j) {
                    const double* center = centers + j * d;
                    double d2 = 0.0;
                    for (py::ssize_t q = 0; q < d; ++q) {
                        const double diff = xi[q] - center[q];
                        d2 += diff * diff;
                    }
                    if (d2 < best) {
                        best = d2;
                        best_index = j;
                    }
                }
                label_data[i] = static_cast<std::int64_t>(best_index);
                distance_data[i] = best;
            }
        });
    }
    return py::make_tuple(labels, distances);
}

py::tuple kmeans_update(Matrix X_, Labels labels_, py::ssize_t n_clusters) {
    const auto Xb = require_matrix(X_, "X");
    if (n_clusters < 1) {
        throw std::invalid_argument("n_clusters must be positive");
    }
    const auto lb = labels_.request();
    if (lb.ndim != 1 || lb.shape[0] != Xb.shape[0]) {
        throw std::invalid_argument("labels must match the number of samples");
    }
    const auto* X = static_cast<const double*>(Xb.ptr);
    const auto* labels = static_cast<const std::int64_t*>(lb.ptr);
    const py::ssize_t n = Xb.shape[0];
    const py::ssize_t d = Xb.shape[1];
    py::array_t<double> sums({n_clusters, d});
    py::array_t<std::int64_t> counts(n_clusters);
    auto* sum_data = static_cast<double*>(sums.request().ptr);
    auto* count_data = static_cast<std::int64_t*>(counts.request().ptr);
    std::fill(sum_data, sum_data + n_clusters * d, 0.0);
    std::fill(count_data, count_data + n_clusters, 0);
    for (py::ssize_t i = 0; i < n; ++i) {
        const std::int64_t cluster = labels[i];
        if (cluster < 0 || cluster >= n_clusters) {
            throw std::invalid_argument("labels must be between 0 and n_clusters - 1");
        }
        ++count_data[cluster];
        for (py::ssize_t q = 0; q < d; ++q) {
            sum_data[cluster * d + q] += X[i * d + q];
        }
    }
    return py::make_tuple(sums, counts);
}

// ---------------------------------------------------------------------------
// GMM E-step and M-step.
// ---------------------------------------------------------------------------

py::array_t<double> gmm_log_gaussian(
    Matrix X_, Matrix means_, Matrix precisions_, Vector log_determinants_) {
    const auto Xb = require_matrix(X_, "X");
    const auto Mb = require_matrix(means_, "means");
    const auto Pb = precisions_.request();
    require_same_features(Xb, Mb);
    if (Pb.ndim != 3 || Pb.shape[0] != Mb.shape[0] || Pb.shape[1] != Mb.shape[1]
        || Pb.shape[2] != Mb.shape[1]) {
        throw std::invalid_argument("precisions must have shape (n_components, n_features, n_features)");
    }
    const auto ldb = require_vector(log_determinants_, "log_determinants", Mb.shape[0]);
    const auto* X = static_cast<const double*>(Xb.ptr);
    const auto* means = static_cast<const double*>(Mb.ptr);
    const auto* precisions = static_cast<const double*>(Pb.ptr);
    const auto* logdet = static_cast<const double*>(ldb.ptr);
    for (py::ssize_t i = 0; i < Pb.shape[0] * Pb.shape[1] * Pb.shape[2]; ++i) {
        if (!std::isfinite(precisions[i])) {
            throw std::invalid_argument("precisions must contain only finite values");
        }
    }
    const py::ssize_t n = Xb.shape[0];
    const py::ssize_t k = Mb.shape[0];
    const py::ssize_t d = Xb.shape[1];
    py::array_t<double> out({n, k});
    auto* result = static_cast<double*>(out.request().ptr);
    constexpr double pi = 3.141592653589793238462643383279502884;
    const double constant = d * std::log(2.0 * pi);
    {
        py::gil_scoped_release release;
        parallel_rows(n, k, [&](py::ssize_t begin, py::ssize_t end) {
            std::vector<double> diff(static_cast<std::size_t>(d));
            for (py::ssize_t i = begin; i < end; ++i) {
                const double* xi = X + i * d;
                for (py::ssize_t j = 0; j < k; ++j) {
                    const double* mean = means + j * d;
                    const double* precision = precisions + j * d * d;
                    for (py::ssize_t q = 0; q < d; ++q) {
                        diff[static_cast<std::size_t>(q)] = xi[q] - mean[q];
                    }
                    double quad = 0.0;
                    for (py::ssize_t q = 0; q < d; ++q) {
                        double projected = 0.0;
                        for (py::ssize_t r = 0; r < d; ++r) {
                            projected += precision[q * d + r] * diff[static_cast<std::size_t>(r)];
                        }
                        quad += diff[static_cast<std::size_t>(q)] * projected;
                    }
                    result[i * k + j] = -0.5 * (constant + logdet[j] + quad);
                }
            }
        });
    }
    return out;
}

py::tuple gmm_m_step(Matrix X_, Matrix responsibilities_, double reg_covar) {
    if (!std::isfinite(reg_covar) || reg_covar < 0.0) {
        throw std::invalid_argument("reg_covar must be non-negative and finite");
    }
    const auto Xb = require_matrix(X_, "X");
    const auto Rb = require_matrix(responsibilities_, "responsibilities");
    if (Rb.shape[0] != Xb.shape[0]) {
        throw std::invalid_argument("responsibilities must match the number of samples");
    }
    const auto* X = static_cast<const double*>(Xb.ptr);
    const auto* R = static_cast<const double*>(Rb.ptr);
    const py::ssize_t n = Xb.shape[0];
    const py::ssize_t d = Xb.shape[1];
    const py::ssize_t k = Rb.shape[1];
    py::array_t<double> weights(k);
    py::array_t<double> means({k, d});
    py::array_t<double> covariances({k, d, d});
    auto* W = static_cast<double*>(weights.request().ptr);
    auto* M = static_cast<double*>(means.request().ptr);
    auto* C = static_cast<double*>(covariances.request().ptr);
    std::vector<double> nk(static_cast<std::size_t>(k), 0.0);
    std::fill(M, M + k * d, 0.0);
    std::fill(C, C + k * d * d, 0.0);
    for (py::ssize_t i = 0; i < n; ++i) {
        for (py::ssize_t j = 0; j < k; ++j) {
            const double responsibility = R[i * k + j];
            if (responsibility < 0.0 || !std::isfinite(responsibility)) {
                throw std::invalid_argument("responsibilities must be finite and non-negative");
            }
            nk[static_cast<std::size_t>(j)] += responsibility;
            for (py::ssize_t q = 0; q < d; ++q) {
                M[j * d + q] += responsibility * X[i * d + q];
            }
        }
    }
    constexpr double eps = 1e-12;
    for (py::ssize_t j = 0; j < k; ++j) {
        const double denom = nk[static_cast<std::size_t>(j)] + eps;
        W[j] = denom / static_cast<double>(n);
        for (py::ssize_t q = 0; q < d; ++q) {
            M[j * d + q] /= denom;
        }
    }
    for (py::ssize_t i = 0; i < n; ++i) {
        for (py::ssize_t j = 0; j < k; ++j) {
            const double responsibility = R[i * k + j];
            for (py::ssize_t q = 0; q < d; ++q) {
                const double dq = X[i * d + q] - M[j * d + q];
                for (py::ssize_t r = 0; r < d; ++r) {
                    const double dr = X[i * d + r] - M[j * d + r];
                    C[j * d * d + q * d + r] += responsibility * dq * dr;
                }
            }
        }
    }
    for (py::ssize_t j = 0; j < k; ++j) {
        const double denom = nk[static_cast<std::size_t>(j)] + eps;
        for (py::ssize_t q = 0; q < d; ++q) {
            for (py::ssize_t r = 0; r < d; ++r) {
                C[j * d * d + q * d + r] /= denom;
            }
            C[j * d * d + q * d + q] += reg_covar;
        }
    }
    return py::make_tuple(weights, means, covariances);
}

PYBIND11_MODULE(_ml_kernels_cpp, m) {
    m.doc() = "C++ accelerated machine-learning kernels";
    m.def("rbf_affinity", &rbf_affinity);
    m.def("knn_affinity", &knn_affinity);
    m.def("knn_graph", &knn_graph);
    m.def("knn_impute", &knn_impute);
    m.def("radius_neighbors", &radius_neighbors);
    m.def("dbscan_neighbors", &dbscan_neighbors);
    m.def("dbscan_labels", &dbscan_labels);
    m.def("meanshift_update", &meanshift_update);
    m.def("hmm_forward", &hmm_forward);
    m.def("hmm_backward", &hmm_backward);
    m.def("hmm_viterbi", &hmm_viterbi);
    m.def("csr_propagate", &csr_propagate);
    m.def("kmeans_assign", &kmeans_assign);
    m.def("kmeans_update", &kmeans_update);
    m.def("gmm_log_gaussian", &gmm_log_gaussian);
    m.def("gmm_m_step", &gmm_m_step);
}
