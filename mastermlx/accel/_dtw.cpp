#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <cmath>
#include <algorithm>

extern "C" {

static PyObject* dtw_path(PyObject*, PyObject* args) {
    PyArrayObject *X_arr, *Y_arr;
    int window;
    if (!PyArg_ParseTuple(args, "O!O!i", &PyArray_Type, &X_arr, &PyArray_Type, &Y_arr, &window))
        return nullptr;

    int n = (int)PyArray_DIM(X_arr, 0);
    int m = (int)PyArray_DIM(Y_arr, 0);
    double* x = (double*)PyArray_DATA(X_arr);
    double* y = (double*)PyArray_DATA(Y_arr);

    if (window < abs(n - m)) window = abs(n - m);

    double* dp = new double[(n + 1) * (m + 1)];
    std::fill_n(dp, (n + 1) * (m + 1), 1e308);
    dp[0] = 0.0;

    for (int i = 1; i <= n; i++) {
        int j_start = std::max(1, i - window);
        int j_end = std::min(m, i + window);
        for (int j = j_start; j <= j_end; j++) {
            double cost = std::abs(x[i - 1] - y[j - 1]);
            double d1 = dp[(i - 1) * (m + 1) + j];
            double d2 = dp[i * (m + 1) + (j - 1)];
            double d3 = dp[(i - 1) * (m + 1) + (j - 1)];
            dp[i * (m + 1) + j] = cost + std::min({d1, d2, d3});
        }
    }

    // Backtrack path
    int path_len = n + m;
    int* path_i = new int[path_len];
    int* path_j = new int[path_len];
    int pi = n, pj = m, k = 0;
    while (pi > 0 && pj > 0) {
        path_i[k] = pi - 1;
        path_j[k] = pj - 1;
        k++;
        double d1 = dp[(pi - 1) * (m + 1) + pj];
        double d2 = dp[pi * (m + 1) + (pj - 1)];
        double d3 = dp[(pi - 1) * (m + 1) + (pj - 1)];
        if (d1 <= d2 && d1 <= d3) pi--;
        else if (d2 <= d1 && d2 <= d3) pj--;
        else { pi--; pj--; }
    }

    double dist = dp[n * (m + 1) + m];
    delete[] dp;

    npy_intp dims[2] = {k, 2};
    PyObject* path_arr = PyArray_SimpleNew(2, dims, NPY_INT);
    int* out_path = (int*)PyArray_DATA((PyArrayObject*)path_arr);
    for (int t = 0; t < k; t++) {
        out_path[(k - 1 - t) * 2] = path_i[t];
        out_path[(k - 1 - t) * 2 + 1] = path_j[t];
    }
    delete[] path_i;
    delete[] path_j;

    PyObject* result = PyTuple_New(2);
    PyTuple_SetItem(result, 0, path_arr);
    PyTuple_SetItem(result, 1, PyFloat_FromDouble(dist));
    return result;
}

static PyMethodDef methods[] = {
    {"dtw_path", dtw_path, METH_VARARGS, "DTW path and distance"},
    {nullptr, nullptr, 0, nullptr}
};
static struct PyModuleDef module = { PyModuleDef_HEAD_INIT, "_dtw", nullptr, -1, methods };
PyMODINIT_FUNC PyInit__dtw(void) { import_array(); return PyModule_Create(&module); }
}
