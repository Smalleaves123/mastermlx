from __future__ import annotations

import importlib

import numpy as np


def _load_backend():
    try:
        return importlib.import_module("mastermlx.accel._cnn_ops")
    except ImportError:
        return None


def _numpy_im2col(X, kh, kw, stride=1, pad=0):
    X = np.asarray(X, dtype=float)
    N, H, W, C = X.shape
    if pad > 0:
        X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
    else:
        X_pad = X
    OH = (H + 2 * pad - kh) // stride + 1
    OW = (W + 2 * pad - kw) // stride + 1
    cols = np.empty((N, OH, OW, kh, kw, C), dtype=float)
    for i in range(kh):
        for j in range(kw):
            cols[:, :, :, i, j, :] = X_pad[:, i:i + OH * stride:stride, j:j + OW * stride:stride, :]
    return cols.reshape(N * OH * OW, kh * kw * C), OH, OW


def _numpy_col2im(cols, shape, kh, kw, stride=1, pad=0):
    cols = np.asarray(cols, dtype=float)
    N, H, W, C = shape
    OH = (H + 2 * pad - kh) // stride + 1
    OW = (W + 2 * pad - kw) // stride + 1
    cols_6d = cols.reshape(N, OH, OW, kh, kw, C)
    dX = np.zeros((N, H + 2 * pad, W + 2 * pad, C), dtype=float)
    for i in range(kh):
        for j in range(kw):
            dX[:, i:i + OH * stride:stride, j:j + OW * stride:stride, :] += cols_6d[:, :, :, i, j, :]
    if pad > 0:
        return dX[:, pad:-pad, pad:-pad, :]
    return dX


def _numpy_maxpool_forward(X, k, stride):
    X = np.asarray(X, dtype=float)
    N, H, W, C = X.shape
    OH = (H - k) // stride + 1
    OW = (W - k) // stride + 1
    out = np.empty((N, OH, OW, C), dtype=float)
    argmax = np.empty((N, OH, OW, C), dtype=np.intp)
    for i in range(OH):
        for j in range(OW):
            patch = X[:, i * stride:i * stride + k, j * stride:j * stride + k, :]
            flat = patch.reshape(N, k * k, C)
            idx = np.argmax(flat, axis=1)
            out[:, i, j, :] = np.take_along_axis(flat, idx[:, None, :], axis=1).squeeze(1)
            argmax[:, i, j, :] = idx
    return out, argmax


def _numpy_maxpool_backward(grad, argmax, shape, k, stride):
    grad = np.asarray(grad, dtype=float)
    argmax = np.asarray(argmax, dtype=np.intp)
    N, H, W, C = shape
    OH, OW = grad.shape[1], grad.shape[2]
    dX = np.zeros(shape, dtype=float)
    for i in range(OH):
        for j in range(OW):
            idx = argmax[:, i, j, :]
            bi = idx // k
            bj = idx % k
            for n in range(N):
                for c in range(C):
                    dX[n, i * stride + bi[n, c], j * stride + bj[n, c], c] += grad[n, i, j, c]
    return dX


def im2col(X, kh, kw, stride=1, pad=0):
    mod = _load_backend()
    if mod is not None:
        return mod.im2col(X, kh, kw, stride, pad)
    return _numpy_im2col(X, kh, kw, stride, pad)


def col2im(cols, shape, kh, kw, stride=1, pad=0):
    mod = _load_backend()
    if mod is not None:
        return mod.col2im(cols, shape, kh, kw, stride, pad)
    return _numpy_col2im(cols, shape, kh, kw, stride, pad)


def maxpool_forward(X, k, stride):
    mod = _load_backend()
    if mod is not None:
        return mod.maxpool_forward(X, k, stride)
    return _numpy_maxpool_forward(X, k, stride)


def maxpool_backward(grad, argmax, shape, k, stride):
    mod = _load_backend()
    if mod is not None:
        return mod.maxpool_backward(grad, argmax, shape, k, stride)
    return _numpy_maxpool_backward(grad, argmax, shape, k, stride)
