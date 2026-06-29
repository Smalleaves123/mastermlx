#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cmath>
#include <algorithm>
#include <queue>
#include <vector>
namespace py = pybind11;

struct Node {
    int idx, dim; double val;
    Node *left, *right;
    Node() : idx(-1), dim(0), val(0), left(nullptr), right(nullptr) {}
};

static Node* build(double* data, int* idxs, int n, int d, int depth) {
    if (n == 0) return nullptr;
    if (n == 1 || depth > 40) { Node* leaf = new Node(); leaf->idx = idxs[0]; return leaf; }
    int dim = depth % d;
    std::sort(idxs, idxs + n, [&](int a, int b) { return data[a*d + dim] < data[b*d + dim]; });
    int mid = n / 2;
    Node* node = new Node();
    node->dim = dim; node->val = data[idxs[mid]*d + dim]; node->idx = idxs[mid];
    node->left = build(data, idxs, mid, d, depth + 1);
    node->right = build(data, idxs + mid + 1, n - mid - 1, d, depth + 1);
    return node;
}

static void destroy(Node* n) { if (n) { destroy(n->left); destroy(n->right); delete n; } }

static double sq(double* a, double* b, int d) {
    double s = 0; for (int i = 0; i < d; i++) { double df = a[i]-b[i]; s += df*df; } return s;
}

struct HeapItem { double dist; int idx; bool operator<(const HeapItem& o) const { return dist < o.dist; } };

static void knn(Node* node, double* q, double* data, int d, int k, std::priority_queue<HeapItem>& heap) {
    if (!node) return;
    if (node->idx >= 0) {
        double dist = sq(q, data + node->idx * d, d);
        if ((int)heap.size() < k) heap.push({dist, node->idx});
        else if (dist < heap.top().dist) { heap.pop(); heap.push({dist, node->idx}); }
    }
    if (node->left || node->right) {
        double diff = q[node->dim] - node->val;
        Node *near = diff <= 0 ? node->left : node->right, *far = diff <= 0 ? node->right : node->left;
        knn(near, q, data, d, k, heap);
        double best = heap.empty() ? 1e308 : heap.top().dist;
        if (diff * diff < best || (int)heap.size() < k) knn(far, q, data, d, k, heap);
    }
}

class KDTree {
    Node* root; double* data; int n, d;
public:
    KDTree(py::array_t<double> X) {
        py::buffer_info buf = X.request(); n = (int)buf.shape[0]; d = (int)buf.shape[1];
        data = (double*)buf.ptr;
        std::vector<int> idxs(n);
        for (int i = 0; i < n; i++) idxs[i] = i;
        root = build(data, idxs.data(), n, d, 0);
    }
    ~KDTree() { destroy(root); }

    std::pair<py::array_t<int>, py::array_t<double>> query(py::array_t<double> Q, int k) {
        py::buffer_info buf = Q.request(); int nq = (int)buf.shape[0];
        double* queries = (double*)buf.ptr;
        k = std::min(k, n);
        py::array_t<int> idx_arr({nq, k});
        py::array_t<double> dst_arr({nq, k});
        py::buffer_info ib = idx_arr.request(), db = dst_arr.request();
        int* oi = (int*)ib.ptr;
        double* od = (double*)db.ptr;
        for (int i = 0; i < nq; i++) {
            std::priority_queue<HeapItem> heap;
            knn(root, queries + i * d, data, d, k, heap);
            int pos = (int)heap.size() - 1;
            while (!heap.empty()) { oi[i*k + pos] = heap.top().idx; od[i*k + pos] = heap.top().dist; heap.pop(); pos--; }
        }
        return {idx_arr, dst_arr};
    }
};

PYBIND11_MODULE(_kdtree, m) {
    py::class_<KDTree>(m, "KDTree")
        .def(py::init<py::array_t<double>>())
        .def("query", &KDTree::query);
}
