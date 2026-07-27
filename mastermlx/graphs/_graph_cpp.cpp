#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <queue>
#include <stdexcept>
#include <stack>
#include <vector>

namespace py = pybind11;
using IntArray = py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>;
using FloatArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

struct CsrView {
    const std::int64_t* indptr;
    const std::int64_t* indices;
    std::int64_t nodes;
    std::int64_t edges;
};

CsrView validate_csr(const IntArray& indptr_, const IntArray& indices_) {
    const auto indptr = indptr_.request();
    const auto indices = indices_.request();
    if (indptr.ndim != 1 || indptr.shape[0] < 1) {
        throw std::invalid_argument("indptr must be a non-empty 1D array");
    }
    if (indices.ndim != 1) {
        throw std::invalid_argument("indices must be a 1D array");
    }

    const auto* offsets = static_cast<const std::int64_t*>(indptr.ptr);
    const auto* neighbors = static_cast<const std::int64_t*>(indices.ptr);
    const std::int64_t nodes = static_cast<std::int64_t>(indptr.shape[0] - 1);
    const std::int64_t edges = static_cast<std::int64_t>(indices.shape[0]);
    if (offsets[0] != 0 || offsets[nodes] != edges) {
        throw std::invalid_argument("indptr must start at zero and end at len(indices)");
    }
    for (std::int64_t i = 0; i < nodes; ++i) {
        if (offsets[i] > offsets[i + 1] || offsets[i] < 0) {
            throw std::invalid_argument("indptr must be non-decreasing and non-negative");
        }
    }
    for (std::int64_t i = 0; i < edges; ++i) {
        if (neighbors[i] < 0 || neighbors[i] >= nodes) {
            throw std::invalid_argument("indices contain a node outside the graph");
        }
    }
    return {offsets, neighbors, nodes, edges};
}

const double* validate_weights(const FloatArray& weights_, const CsrView& csr) {
    const auto weights = weights_.request();
    if (weights.ndim != 1 || weights.shape[0] != csr.edges) {
        throw std::invalid_argument("weights must be a 1D array matching indices");
    }
    const auto* values = static_cast<const double*>(weights.ptr);
    for (std::int64_t edge = 0; edge < csr.edges; ++edge) {
        if (!std::isfinite(values[edge]) || values[edge] < 0.0) {
            throw std::invalid_argument("weights must be finite and non-negative");
        }
    }
    return values;
}

py::array_t<std::int64_t> bfs_order(IntArray indptr_, IntArray indices_, std::int64_t start) {
    const auto csr = validate_csr(indptr_, indices_);
    if (start < 0 || start >= csr.nodes) {
        throw std::invalid_argument("start must be a valid node id");
    }
    std::vector<std::int64_t> order(static_cast<std::size_t>(csr.nodes));
    std::vector<unsigned char> visited(static_cast<std::size_t>(csr.nodes), 0);
    std::queue<std::int64_t> queue;
    std::int64_t count = 0;
    {
        py::gil_scoped_release release;
        visited[static_cast<std::size_t>(start)] = 1;
        queue.push(start);
        while (!queue.empty()) {
            const auto node = queue.front();
            queue.pop();
            order[static_cast<std::size_t>(count++)] = node;
            for (std::int64_t edge = csr.indptr[node]; edge < csr.indptr[node + 1]; ++edge) {
                const auto neighbor = csr.indices[edge];
                if (visited[static_cast<std::size_t>(neighbor)] == 0) {
                    visited[static_cast<std::size_t>(neighbor)] = 1;
                    queue.push(neighbor);
                }
            }
        }
    }
    py::array_t<std::int64_t> output(count);
    auto* output_data = static_cast<std::int64_t*>(output.request().ptr);
    std::copy(order.begin(), order.begin() + count, output_data);
    return output;
}

py::array_t<std::int64_t> connected_components(IntArray indptr_, IntArray indices_) {
    const auto csr = validate_csr(indptr_, indices_);
    py::array_t<std::int64_t> labels(csr.nodes);
    auto* output = static_cast<std::int64_t*>(labels.request().ptr);
    std::vector<std::int64_t> parent(static_cast<std::size_t>(csr.nodes));
    std::vector<std::int64_t> rank(static_cast<std::size_t>(csr.nodes), 0);
    {
        py::gil_scoped_release release;
        for (std::int64_t node = 0; node < csr.nodes; ++node) {
            parent[static_cast<std::size_t>(node)] = node;
        }
        auto find = [&parent](std::int64_t node) {
            std::int64_t root = node;
            while (parent[static_cast<std::size_t>(root)] != root) {
                root = parent[static_cast<std::size_t>(root)];
            }
            while (parent[static_cast<std::size_t>(node)] != node) {
                const auto next = parent[static_cast<std::size_t>(node)];
                parent[static_cast<std::size_t>(node)] = root;
                node = next;
            }
            return root;
        };
        auto unite = [&parent, &rank, &find](std::int64_t left, std::int64_t right) {
            left = find(left);
            right = find(right);
            if (left == right) {
                return;
            }
            if (rank[static_cast<std::size_t>(left)] < rank[static_cast<std::size_t>(right)]) {
                std::swap(left, right);
            }
            parent[static_cast<std::size_t>(right)] = left;
            if (rank[static_cast<std::size_t>(left)] == rank[static_cast<std::size_t>(right)]) {
                ++rank[static_cast<std::size_t>(left)];
            }
        };
        for (std::int64_t node = 0; node < csr.nodes; ++node) {
            for (std::int64_t edge = csr.indptr[node]; edge < csr.indptr[node + 1]; ++edge) {
                unite(node, csr.indices[edge]);
            }
        }
        for (std::int64_t node = 0; node < csr.nodes; ++node) {
            output[node] = find(node);
        }
    }
    return labels;
}

py::array_t<std::int64_t> topological_order(IntArray indptr_, IntArray indices_) {
    const auto csr = validate_csr(indptr_, indices_);
    py::array_t<std::int64_t> output(csr.nodes);
    auto* order = static_cast<std::int64_t*>(output.request().ptr);
    std::vector<std::int64_t> indegree(static_cast<std::size_t>(csr.nodes), 0);
    std::priority_queue<std::int64_t, std::vector<std::int64_t>, std::greater<>> ready;
    std::int64_t count = 0;
    {
        py::gil_scoped_release release;
        for (std::int64_t node = 0; node < csr.nodes; ++node) {
            for (std::int64_t edge = csr.indptr[node]; edge < csr.indptr[node + 1]; ++edge) {
                ++indegree[static_cast<std::size_t>(csr.indices[edge])];
            }
        }
        for (std::int64_t node = 0; node < csr.nodes; ++node) {
            if (indegree[static_cast<std::size_t>(node)] == 0) {
                ready.push(node);
            }
        }
        while (!ready.empty()) {
            const auto node = ready.top();
            ready.pop();
            order[count++] = node;
            for (std::int64_t edge = csr.indptr[node]; edge < csr.indptr[node + 1]; ++edge) {
                const auto neighbor = csr.indices[edge];
                if (--indegree[static_cast<std::size_t>(neighbor)] == 0) {
                    ready.push(neighbor);
                }
            }
        }
    }
    if (count != csr.nodes) {
        throw std::invalid_argument("graph contains a directed cycle");
    }
    return output;
}

py::tuple dijkstra_weighted(
    IntArray indptr_,
    IntArray indices_,
    FloatArray weights_,
    std::int64_t start,
    std::int64_t goal
) {
    const auto csr = validate_csr(indptr_, indices_);
    const auto* weights = validate_weights(weights_, csr);
    if (start < 0 || start >= csr.nodes) {
        throw std::invalid_argument("start must be a valid node id");
    }
    if (goal < -1 || goal >= csr.nodes) {
        throw std::invalid_argument("goal must be -1 or a valid node id");
    }

    const double inf = std::numeric_limits<double>::infinity();
    std::vector<double> distances(static_cast<std::size_t>(csr.nodes), inf);
    std::vector<std::int64_t> predecessors(static_cast<std::size_t>(csr.nodes), -1);
    using QueueItem = std::pair<double, std::int64_t>;
    std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<QueueItem>> queue;
    distances[static_cast<std::size_t>(start)] = 0.0;
    queue.emplace(0.0, start);

    {
        py::gil_scoped_release release;
        while (!queue.empty()) {
            const QueueItem current = queue.top();
            queue.pop();
            const double cost = current.first;
            const auto node = current.second;
            if (cost > distances[static_cast<std::size_t>(node)]) {
                continue;
            }
            if (node == goal) {
                break;
            }
            for (std::int64_t edge = csr.indptr[node]; edge < csr.indptr[node + 1]; ++edge) {
                const auto neighbor = csr.indices[edge];
                const double next_cost = cost + weights[edge];
                auto& distance = distances[static_cast<std::size_t>(neighbor)];
                if (next_cost < distance) {
                    distance = next_cost;
                    predecessors[static_cast<std::size_t>(neighbor)] = node;
                    queue.emplace(next_cost, neighbor);
                }
            }
        }
    }

    py::array_t<double> distance_output(csr.nodes);
    py::array_t<std::int64_t> predecessor_output(csr.nodes);
    std::copy(distances.begin(), distances.end(), static_cast<double*>(distance_output.request().ptr));
    std::copy(
        predecessors.begin(),
        predecessors.end(),
        static_cast<std::int64_t*>(predecessor_output.request().ptr)
    );
    return py::make_tuple(distance_output, predecessor_output);
}

py::array_t<std::int64_t> multi_source_bfs_distances(
    IntArray indptr_,
    IntArray indices_,
    IntArray starts_
) {
    const auto csr = validate_csr(indptr_, indices_);
    const auto starts = starts_.request();
    if (starts.ndim != 1) {
        throw std::invalid_argument("starts must be a 1D array");
    }
    const auto* sources = static_cast<const std::int64_t*>(starts.ptr);
    std::vector<std::int64_t> distances(static_cast<std::size_t>(csr.nodes), -1);
    std::queue<std::int64_t> queue;
    for (py::ssize_t index = 0; index < starts.shape[0]; ++index) {
        const auto source = sources[index];
        if (source < 0 || source >= csr.nodes) {
            throw std::invalid_argument("starts contains an invalid node id");
        }
        if (distances[static_cast<std::size_t>(source)] == -1) {
            distances[static_cast<std::size_t>(source)] = 0;
            queue.push(source);
        }
    }

    {
        py::gil_scoped_release release;
        while (!queue.empty()) {
            const auto node = queue.front();
            queue.pop();
            const auto next_distance = distances[static_cast<std::size_t>(node)] + 1;
            for (std::int64_t edge = csr.indptr[node]; edge < csr.indptr[node + 1]; ++edge) {
                const auto neighbor = csr.indices[edge];
                auto& distance = distances[static_cast<std::size_t>(neighbor)];
                if (distance == -1) {
                    distance = next_distance;
                    queue.push(neighbor);
                }
            }
        }
    }

    py::array_t<std::int64_t> output(csr.nodes);
    std::copy(distances.begin(), distances.end(), static_cast<std::int64_t*>(output.request().ptr));
    return output;
}

py::array_t<std::int64_t> strongly_connected_components(IntArray indptr_, IntArray indices_) {
    const auto csr = validate_csr(indptr_, indices_);
    std::vector<std::vector<std::int64_t>> reverse(static_cast<std::size_t>(csr.nodes));
    std::vector<unsigned char> visited(static_cast<std::size_t>(csr.nodes), 0);
    std::vector<std::int64_t> finish_order;
    std::vector<std::int64_t> labels(static_cast<std::size_t>(csr.nodes), -1);
    finish_order.reserve(static_cast<std::size_t>(csr.nodes));

    {
        py::gil_scoped_release release;
        for (std::int64_t root = 0; root < csr.nodes; ++root) {
            if (visited[static_cast<std::size_t>(root)] != 0) {
                continue;
            }
            std::vector<std::pair<std::int64_t, std::int64_t>> stack;
            visited[static_cast<std::size_t>(root)] = 1;
            stack.emplace_back(root, csr.indptr[root]);
            while (!stack.empty()) {
                auto& frame = stack.back();
                const auto node = frame.first;
                auto& edge = frame.second;
                if (edge < csr.indptr[node + 1]) {
                    const auto neighbor = csr.indices[edge++];
                    reverse[static_cast<std::size_t>(neighbor)].push_back(node);
                    if (visited[static_cast<std::size_t>(neighbor)] == 0) {
                        visited[static_cast<std::size_t>(neighbor)] = 1;
                        stack.emplace_back(neighbor, csr.indptr[neighbor]);
                    }
                } else {
                    finish_order.push_back(node);
                    stack.pop_back();
                }
            }
        }

        std::int64_t component = 0;
        for (auto it = finish_order.rbegin(); it != finish_order.rend(); ++it) {
            const auto root = *it;
            if (labels[static_cast<std::size_t>(root)] != -1) {
                continue;
            }
            std::vector<std::int64_t> stack = {root};
            labels[static_cast<std::size_t>(root)] = component;
            while (!stack.empty()) {
                const auto node = stack.back();
                stack.pop_back();
                for (const auto neighbor : reverse[static_cast<std::size_t>(node)]) {
                    auto& label = labels[static_cast<std::size_t>(neighbor)];
                    if (label == -1) {
                        label = component;
                        stack.push_back(neighbor);
                    }
                }
            }
            ++component;
        }

    }

    py::array_t<std::int64_t> output(csr.nodes);
    std::copy(labels.begin(), labels.end(), static_cast<std::int64_t*>(output.request().ptr));
    return output;
}

PYBIND11_MODULE(_graph_cpp, m) {
    m.doc() = "C++ kernels for CSR graph traversal and analysis";
    m.def("bfs_order", &bfs_order, py::arg("indptr"), py::arg("indices"), py::arg("start"));
    m.def("connected_components", &connected_components, py::arg("indptr"), py::arg("indices"));
    m.def("topological_order", &topological_order, py::arg("indptr"), py::arg("indices"));
    m.def(
        "dijkstra_weighted",
        &dijkstra_weighted,
        py::arg("indptr"),
        py::arg("indices"),
        py::arg("weights"),
        py::arg("start"),
        py::arg("goal")
    );
    m.def(
        "multi_source_bfs_distances",
        &multi_source_bfs_distances,
        py::arg("indptr"),
        py::arg("indices"),
        py::arg("starts")
    );
    m.def(
        "strongly_connected_components",
        &strongly_connected_components,
        py::arg("indptr"),
        py::arg("indices")
    );
}
