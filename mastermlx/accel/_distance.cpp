#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <cmath>
#include <algorithm>

// ---------------------------------------------------------------------------
// pairwise_squared_euclidean
// ---------------------------------------------------------------------------
static PyObject* cpp_pairwise_sq_euclid(PyObject*, PyObject* args) {
    PyArrayObject *X_arr = nullptr, *Y_arr = nullptr;
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &X_arr, &PyArray_Type, &Y_arr))
        return nullptr;

    if (PyArray_NDIM(X_arr) != 2 || PyArray_NDIM(Y_arr) != 2) {
        PyErr_SetString(PyExc_ValueError, "X and Y must be 2D");
        return nullptr;
    }

    npy_intp n = PyArray_DIM(X_arr, 0);
    npy_intp m = PyArray_DIM(Y_arr, 0);
    npy_intp d = PyArray_DIM(X_arr, 1);

    if (PyArray_DIM(Y_arr, 1) != d) {
        PyErr_SetString(PyExc_ValueError, "X and Y must have same feature count");
        return nullptr;
    }

    npy_intp dims[2] = {n, m};
    PyObject* out = PyArray_SimpleNew(2, dims, NPY_FLOAT64);
    if (!out) return nullptr;
    double* out_data = (double*)PyArray_DATA((PyArrayObject*)out);

    double* X = (double*)PyArray_DATA(X_arr);
    double* Y = (double*)PyArray_DATA(Y_arr);

    #pragma omp parallel for collapse(2) if(n * m > 10000)
    for (npy_intp i = 0; i < n; i++) {
        for (npy_intp j = 0; j < m; j++) {
            double acc = 0.0;
            double* xi = X + i * d;
            double* yj = Y + j * d;
            for (npy_intp k = 0; k < d; k++) {
                double diff = xi[k] - yj[k];
                acc += diff * diff;
            }
            out_data[i * m + j] = acc;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// pairwise_distances (Euclidean)
// ---------------------------------------------------------------------------
static PyObject* cpp_pairwise_dist(PyObject*, PyObject* args) {
    PyArrayObject *X_arr = nullptr, *Y_arr = nullptr;
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &X_arr, &PyArray_Type, &Y_arr))
        return nullptr;

    npy_intp n = PyArray_DIM(X_arr, 0), m = PyArray_DIM(Y_arr, 0), d = PyArray_DIM(X_arr, 1);
    npy_intp dims[2] = {n, m};
    PyObject* out = PyArray_SimpleNew(2, dims, NPY_FLOAT64);
    double* out_data = (double*)PyArray_DATA((PyArrayObject*)out);
    double* X = (double*)PyArray_DATA(X_arr);
    double* Y = (double*)PyArray_DATA(Y_arr);

    for (npy_intp i = 0; i < n; i++) {
        for (npy_intp j = 0; j < m; j++) {
            double acc = 0.0;
            for (npy_intp k = 0; k < d; k++) {
                double diff = X[i*d + k] - Y[j*d + k];
                acc += diff * diff;
            }
            out_data[i*m + j] = std::sqrt(std::max(acc, 0.0));
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// pairwise_manhattan
// ---------------------------------------------------------------------------
static PyObject* cpp_pairwise_manhattan(PyObject*, PyObject* args) {
    PyArrayObject *X_arr = nullptr, *Y_arr = nullptr;
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &X_arr, &PyArray_Type, &Y_arr))
        return nullptr;

    npy_intp n = PyArray_DIM(X_arr, 0), m = PyArray_DIM(Y_arr, 0), d = PyArray_DIM(X_arr, 1);
    npy_intp dims[2] = {n, m};
    PyObject* out = PyArray_SimpleNew(2, dims, NPY_FLOAT64);
    double* out_data = (double*)PyArray_DATA((PyArrayObject*)out);
    double* X = (double*)PyArray_DATA(X_arr);
    double* Y = (double*)PyArray_DATA(Y_arr);

    for (npy_intp i = 0; i < n; i++) {
        for (npy_intp j = 0; j < m; j++) {
            double acc = 0.0;
            for (npy_intp k = 0; k < d; k++) {
                acc += std::abs(X[i*d + k] - Y[j*d + k]);
            }
            out_data[i*m + j] = acc;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------
static PyMethodDef methods[] = {
    {"pairwise_squared_euclidean", cpp_pairwise_sq_euclid, METH_VARARGS, "Squared Euclidean distances"},
    {"pairwise_distances", cpp_pairwise_dist, METH_VARARGS, "Euclidean distances"},
    {"pairwise_manhattan_distances", cpp_pairwise_manhattan, METH_VARARGS, "Manhattan distances"},
    {nullptr, nullptr, 0, nullptr}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_distance_cpp", nullptr, -1, methods
};

PyMODINIT_FUNC PyInit__distance_cpp(void) {
    import_array();
    return PyModule_Create(&module);
}
