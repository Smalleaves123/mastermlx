#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <queue>
#include <stdexcept>
#include <vector>

namespace py = pybind11;
using IntArray = py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>;

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

PYBIND11_MODULE(_graph_cpp, m) {
    m.doc() = "C++ kernels for CSR graph traversal and analysis";
    m.def("bfs_order", &bfs_order, py::arg("indptr"), py::arg("indices"), py::arg("start"));
    m.def("connected_components", &connected_components, py::arg("indptr"), py::arg("indices"));
    m.def("topological_order", &topological_order, py::arg("indptr"), py::arg("indices"));
}
