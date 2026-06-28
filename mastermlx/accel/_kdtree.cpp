#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <cmath>
#include <algorithm>
#include <queue>
#include <vector>
#include <limits>

struct KDNode {
    int idx;
    int dim;
    double val;
    KDNode *left, *right;
    KDNode() : idx(-1), dim(0), val(0.0), left(nullptr), right(nullptr) {}
};

static KDNode* build_tree(double* data, int* idxs, int n, int d, int depth) {
    if (n == 0) return nullptr;
    if (n == 1 || depth > 40) {
        KDNode* leaf = new KDNode();
        leaf->idx = idxs[0];
        return leaf;
    }
    int dim = depth % d;
    std::sort(idxs, idxs + n, [&](int a, int b) {
        return data[a * d + dim] < data[b * d + dim];
    });
    int mid = n / 2;
    KDNode* node = new KDNode();
    node->dim = dim;
    node->val = data[idxs[mid] * d + dim];
    node->idx = idxs[mid];
    node->left = build_tree(data, idxs, mid, d, depth + 1);
    node->right = build_tree(data, idxs + mid + 1, n - mid - 1, d, depth + 1);

    return node;
}

static void destroy_tree(KDNode* n) {
    if (!n) return;
    destroy_tree(n->left);
    destroy_tree(n->right);
    delete n;
}

static double sq_dist(double* a, double* b, int d) {
    double s = 0.0;
    for (int i = 0; i < d; i++) { double diff = a[i]-b[i]; s += diff*diff; }
    return s;
}

struct HeapItem { double dist; int idx; };
static bool operator<(const HeapItem& a, const HeapItem& b) { return a.dist < b.dist; }

static void knn(KDNode* node, double* query, double* data, int d, int k,
                std::priority_queue<HeapItem>& heap) {
    if (!node) return;

    if (node->left || node->right) {
        // Internal: decide near/far
        double diff = query[node->dim] - node->val;
        KDNode *near = diff <= 0 ? node->left : node->right;
        KDNode *far  = diff <= 0 ? node->right : node->left;
        knn(near, query, data, d, k, heap);
        double best = heap.empty() ? 1e308 : heap.top().dist;
        if (diff * diff < best || (int)heap.size() < k) {
            knn(far, query, data, d, k, heap);
        }
    }


}

extern "C" {

static PyObject* kdtree_new(PyObject*, PyObject* args) {
    PyArrayObject* X = nullptr;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &X)) return nullptr;
    int n = (int)PyArray_DIM(X, 0), d = (int)PyArray_DIM(X, 1);
    double* data = (double*)PyArray_DATA(X);
    std::vector<int> idxs(n);
    for (int i = 0; i < n; i++) idxs[i] = i;
    KDNode* root = build_tree(data, idxs.data(), n, d, 0);
    // Store data pointer + tree in capsule
    void** ptrs = new void*[3];
    ptrs[0] = root;
    ptrs[1] = data;
    ptrs[2] = (void*)(intptr_t)d;
    return PyCapsule_New(ptrs, "kdtree", [](PyObject* cap) {
        void** p = (void**)PyCapsule_GetPointer(cap, "kdtree");
        if (p) { destroy_tree((KDNode*)p[0]); delete[] p; }
    });
}

static PyObject* kdtree_query(PyObject*, PyObject* args) {
    PyObject* cap; PyArrayObject *Q, *karr;
    if (!PyArg_ParseTuple(args, "OO!O!", &cap, &PyArray_Type, &Q, &PyArray_Type, &karr))
        return nullptr;
    void** ptrs = (void**)PyCapsule_GetPointer(cap, "kdtree");
    if (!ptrs) { PyErr_SetString(PyExc_RuntimeError, "bad capsule"); return nullptr; }
    KDNode* root = (KDNode*)ptrs[0];
    double* data = (double*)ptrs[1];
    int d = (int)(intptr_t)ptrs[2];

    int nq = (int)PyArray_DIM(Q, 0);
    double* queries = (double*)PyArray_DATA(Q);
    int k = *(int*)PyArray_DATA(karr);
    k = std::min(k, (int)PyArray_DIM((PyArrayObject*)PyTuple_GetItem(args, 1), 0));

    npy_intp dims[2] = {nq, k};
    PyObject* idx_arr = PyArray_SimpleNew(2, dims, NPY_INT);
    PyObject* dst_arr = PyArray_SimpleNew(2, dims, NPY_FLOAT64);
    int* out_idx = (int*)PyArray_DATA((PyArrayObject*)idx_arr);
    double* out_dst = (double*)PyArray_DATA((PyArrayObject*)dst_arr);

    for (int i = 0; i < nq; i++) {
        std::priority_queue<HeapItem> heap;
        knn(root, queries + i * d, data, d, k, heap);
        int pos = (int)heap.size() - 1;
        while (!heap.empty()) {
            out_idx[i * k + pos] = heap.top().idx;
            out_dst[i * k + pos] = heap.top().dist;
            heap.pop();
            pos--;
        }
        // Fill remaining with -1 if heap has fewer than k
        for (int j = pos; j >= 0; j--) {
            out_idx[i * k + j] = -1;
            out_dst[i * k + j] = 0.0;
        }
    }

    PyObject* result = PyTuple_New(2);
    PyTuple_SetItem(result, 0, idx_arr);
    PyTuple_SetItem(result, 1, dst_arr);
    return result;
}

static PyMethodDef methods[] = {
    {"kdtree_new", kdtree_new, METH_VARARGS, "Build KDTree"},
    {"kdtree_query", kdtree_query, METH_VARARGS, "Query k-NN"},
    {nullptr, nullptr, 0, nullptr}
};
static struct PyModuleDef module = { PyModuleDef_HEAD_INIT, "_kdtree", nullptr, -1, methods };
PyMODINIT_FUNC PyInit__kdtree(void) { import_array(); return PyModule_Create(&module); }
}
