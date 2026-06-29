# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
"""Cython accelerated convolution and max-pooling helpers."""

import numpy as np
cimport numpy as np

ctypedef np.float64_t DTYPE_t
ctypedef np.intp_t ITYPE_t

np.import_array()


def im2col(np.ndarray[DTYPE_t, ndim=4] X, int kh, int kw, int stride=1, int pad=0):
    cdef ITYPE_t N = X.shape[0]
    cdef ITYPE_t H = X.shape[1]
    cdef ITYPE_t W = X.shape[2]
    cdef ITYPE_t C = X.shape[3]
    cdef ITYPE_t OH = (H + 2 * pad - kh) // stride + 1
    cdef ITYPE_t OW = (W + 2 * pad - kw) // stride + 1
    cdef np.ndarray[DTYPE_t, ndim=4] X_pad
    cdef np.ndarray[DTYPE_t, ndim=2] cols
    cdef DTYPE_t[:, :, :, :] Xv
    cdef DTYPE_t[:, :] cv
    cdef ITYPE_t n, oh, ow, i, j, c, row, col

    if pad > 0:
        X_pad = np.pad(np.ascontiguousarray(X, dtype=np.float64),
                       ((0, 0), (pad, pad), (pad, pad), (0, 0)),
                       mode="constant")
    else:
        X_pad = np.ascontiguousarray(X, dtype=np.float64)

    cols = np.empty((N * OH * OW, kh * kw * C), dtype=np.float64)
    Xv = X_pad
    cv = cols

    for n in range(N):
        for oh in range(OH):
            for ow in range(OW):
                row = (n * OH + oh) * OW + ow
                for i in range(kh):
                    for j in range(kw):
                        for c in range(C):
                            col = ((i * kw + j) * C) + c
                            cv[row, col] = Xv[n, oh * stride + i, ow * stride + j, c]

    return cols, OH, OW


def col2im(np.ndarray[DTYPE_t, ndim=2] cols, shape, int kh, int kw, int stride=1, int pad=0):
    cdef ITYPE_t N = shape[0]
    cdef ITYPE_t H = shape[1]
    cdef ITYPE_t W = shape[2]
    cdef ITYPE_t C = shape[3]
    cdef ITYPE_t OH = (H + 2 * pad - kh) // stride + 1
    cdef ITYPE_t OW = (W + 2 * pad - kw) // stride + 1
    cdef np.ndarray[DTYPE_t, ndim=4] dX_pad = np.zeros((N, H + 2 * pad, W + 2 * pad, C), dtype=np.float64)
    cdef DTYPE_t[:, :, :, :] dv = dX_pad
    cdef DTYPE_t[:, :] cv = cols
    cdef ITYPE_t n, oh, ow, i, j, c, row, col

    for n in range(N):
        for oh in range(OH):
            for ow in range(OW):
                row = (n * OH + oh) * OW + ow
                for i in range(kh):
                    for j in range(kw):
                        for c in range(C):
                            col = ((i * kw + j) * C) + c
                            dv[n, oh * stride + i, ow * stride + j, c] += cv[row, col]

    if pad > 0:
        return dX_pad[:, pad:-pad, pad:-pad, :]
    return dX_pad


def maxpool_forward(np.ndarray[DTYPE_t, ndim=4] X, int k, int stride):
    cdef ITYPE_t N = X.shape[0]
    cdef ITYPE_t H = X.shape[1]
    cdef ITYPE_t W = X.shape[2]
    cdef ITYPE_t C = X.shape[3]
    cdef ITYPE_t OH = (H - k) // stride + 1
    cdef ITYPE_t OW = (W - k) // stride + 1
    cdef np.ndarray[DTYPE_t, ndim=4] out = np.empty((N, OH, OW, C), dtype=np.float64)
    cdef np.ndarray[ITYPE_t, ndim=4] argmax = np.empty((N, OH, OW, C), dtype=np.intp)
    cdef DTYPE_t[:, :, :, :] xv = np.ascontiguousarray(X, dtype=np.float64)
    cdef DTYPE_t[:, :, :, :] ov = out
    cdef ITYPE_t[:, :, :, :] av = argmax
    cdef ITYPE_t n, oh, ow, c, i, j, idx, best_idx, base_h, base_w
    cdef DTYPE_t val, best_val

    for n in range(N):
        for oh in range(OH):
            base_h = oh * stride
            for ow in range(OW):
                base_w = ow * stride
                for c in range(C):
                    best_val = xv[n, base_h, base_w, c]
                    best_idx = 0
                    idx = 0
                    for i in range(k):
                        for j in range(k):
                            val = xv[n, base_h + i, base_w + j, c]
                            if idx == 0 or val > best_val:
                                best_val = val
                                best_idx = idx
                            idx += 1
                    ov[n, oh, ow, c] = best_val
                    av[n, oh, ow, c] = best_idx

    return out, argmax


def maxpool_backward(np.ndarray[DTYPE_t, ndim=4] grad, np.ndarray[ITYPE_t, ndim=4] argmax,
                     shape, int k, int stride):
    cdef ITYPE_t N = shape[0]
    cdef ITYPE_t H = shape[1]
    cdef ITYPE_t W = shape[2]
    cdef ITYPE_t C = shape[3]
    cdef ITYPE_t OH = grad.shape[1]
    cdef ITYPE_t OW = grad.shape[2]
    cdef np.ndarray[DTYPE_t, ndim=4] dX = np.zeros((N, H, W, C), dtype=np.float64)
    cdef DTYPE_t[:, :, :, :] gv = np.ascontiguousarray(grad, dtype=np.float64)
    cdef ITYPE_t[:, :, :, :] av = np.ascontiguousarray(argmax, dtype=np.intp)
    cdef DTYPE_t[:, :, :, :] dv = dX
    cdef ITYPE_t n, oh, ow, c, idx, bi, bj, base_h, base_w

    for n in range(N):
        for oh in range(OH):
            base_h = oh * stride
            for ow in range(OW):
                base_w = ow * stride
                for c in range(C):
                    idx = av[n, oh, ow, c]
                    bi = idx // k
                    bj = idx % k
                    dv[n, base_h + bi, base_w + bj, c] += gv[n, oh, ow, c]

    return dX
