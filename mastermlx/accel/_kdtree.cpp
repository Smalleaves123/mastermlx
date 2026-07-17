#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

using Matrix = py::array_t<double, py::array::c_style | py::array::forcecast>;

struct Node {
    int idx;
    int dim;
    double val;
    Node* left;
    Node* right;

    Node() : idx(-1), dim(0), val(0.0), left(nullptr), right(nullptr) {}
};

static Node* build(const double* data, int* idxs, int n, int d, int depth) {
    if (n == 0) {
        return nullptr;
    }
    if (n == 1) {
        Node* leaf = new Node();
        leaf->idx = idxs[0];
        return leaf;
    }
    const int dim = depth % d;
    std::sort(idxs, idxs + n, [&](int a, int b) {
        return data[a * d + dim] < data[b * d + dim];
    });
    const int mid = n / 2;
    Node* node = new Node();
    node->dim = dim;
    node->val = data[idxs[mid] * d + dim];
    node->idx = idxs[mid];
    node->left = build(data, idxs, mid, d, depth + 1);
    node->right = build(data, idxs + mid + 1, n - mid - 1, d, depth + 1);
    return node;
}

static void destroy(Node* node) {
    if (node) {
        destroy(node->left);
        destroy(node->right);
        delete node;
    }
}

static double squared_distance(const double* a, const double* b, int d) {
    double value = 0.0;
    for (int i = 0; i < d; ++i) {
        const double diff = a[i] - b[i];
        value += diff * diff;
    }
    return value;
}

struct HeapItem {
    double dist;
    int idx;

    bool operator<(const HeapItem& other) const {
        return dist < other.dist || (dist == other.dist && idx < other.idx);
    }
};

static void knn(
    Node* node,
    const double* query,
    const double* data,
    int d,
    int k,
    std::priority_queue<HeapItem>& heap) {
    if (!node) {
        return;
    }
    if (node->idx >= 0) {
        const double dist = squared_distance(query, data + node->idx * d, d);
        if (static_cast<int>(heap.size()) < k) {
            heap.push({dist, node->idx});
        } else if (dist < heap.top().dist
                   || (dist == heap.top().dist && node->idx < heap.top().idx)) {
            heap.pop();
            heap.push({dist, node->idx});
        }
    }
    if (node->left || node->right) {
        const double diff = query[node->dim] - node->val;
        Node* near = diff <= 0.0 ? node->left : node->right;
        Node* far = diff <= 0.0 ? node->right : node->left;
        knn(near, query, data, d, k, heap);
        const double best = heap.empty()
            ? std::numeric_limits<double>::infinity()
            : heap.top().dist;
        if (diff * diff < best || static_cast<int>(heap.size()) < k) {
            knn(far, query, data, d, k, heap);
        }
    }
}

class KDTree {
    py::array_t<double> owner_;
    Node* root_;
    const double* data_;
    int n_;
    int d_;

public:
    explicit KDTree(Matrix X)
        : owner_(std::move(X)), root_(nullptr), data_(nullptr), n_(0), d_(0) {
        const auto buf = owner_.request();
        if (buf.ndim != 2) {
            throw std::invalid_argument("X must be a 2D array");
        }
        if (buf.shape[0] <= 0 || buf.shape[1] <= 0) {
            throw std::invalid_argument("X must be non-empty");
        }
        if (buf.shape[0] > std::numeric_limits<int>::max()
            || buf.shape[1] > std::numeric_limits<int>::max()) {
            throw std::invalid_argument("X is too large for the KDTree index type");
        }
        const auto* values = static_cast<const double*>(buf.ptr);
        for (py::ssize_t index = 0; index < buf.size; ++index) {
            if (!std::isfinite(values[index])) {
                throw std::invalid_argument("X must contain only finite values");
            }
        }
        n_ = static_cast<int>(buf.shape[0]);
        d_ = static_cast<int>(buf.shape[1]);
        data_ = static_cast<const double*>(buf.ptr);
        std::vector<int> indices(static_cast<std::size_t>(n_));
        for (int i = 0; i < n_; ++i) {
            indices[static_cast<std::size_t>(i)] = i;
        }
        root_ = build(data_, indices.data(), n_, d_, 0);
    }

    ~KDTree() {
        destroy(root_);
    }

    std::pair<py::array_t<int>, py::array_t<double>> query(Matrix Q, py::ssize_t k) {
        const auto buf = Q.request();
        if (buf.ndim != 2) {
            throw std::invalid_argument("Q must be a 2D array");
        }
        if (buf.shape[1] != d_) {
            throw std::invalid_argument("Q must have the same number of features as X");
        }
        if (buf.shape[0] > std::numeric_limits<int>::max()) {
            throw std::invalid_argument("Q is too large for the KDTree index type");
        }
        if (k < 1 || k > n_) {
            throw std::invalid_argument("k must be between 1 and the number of samples");
        }
        if (k > std::numeric_limits<int>::max()) {
            throw std::invalid_argument("k is too large");
        }
        const int neighbors = static_cast<int>(k);
        const int nq = static_cast<int>(buf.shape[0]);
        const auto* queries = static_cast<const double*>(buf.ptr);
        for (py::ssize_t index = 0; index < buf.size; ++index) {
            if (!std::isfinite(queries[index])) {
                throw std::invalid_argument("Q must contain only finite values");
            }
        }
        py::array_t<int> indices({buf.shape[0], k});
        py::array_t<double> distances({buf.shape[0], k});
        auto* out_indices = static_cast<int*>(indices.request().ptr);
        auto* out_distances = static_cast<double*>(distances.request().ptr);
        {
            py::gil_scoped_release release;
            for (int i = 0; i < nq; ++i) {
                std::priority_queue<HeapItem> heap;
                knn(root_, queries + i * d_, data_, d_, neighbors, heap);
                int position = static_cast<int>(heap.size()) - 1;
                while (!heap.empty()) {
                    out_indices[i * neighbors + position] = heap.top().idx;
                    out_distances[i * neighbors + position] = heap.top().dist;
                    heap.pop();
                    --position;
                }
            }
        }
        return {indices, distances};
    }
};

PYBIND11_MODULE(_kdtree, m) {
    py::class_<KDTree>(m, "KDTree")
        .def(py::init<Matrix>())
        .def("query", &KDTree::query, py::arg("Q"), py::arg("k"));
}
