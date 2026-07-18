from __future__ import annotations

import numpy as np

from ...accel.rnn_ops import gru_forward, lstm_forward, simple_rnn_forward
from ...base import BaseLayer


class SimpleRNN(BaseLayer):
    """Elman RNN: h_t = tanh(W_xh·x_t + W_hh·h_{t-1} + b)."""

    def __init__(self, n_units, return_sequences=False, random_state=None):
        self.n_units = int(n_units)
        self.return_sequences = bool(return_sequences)
        self.random_state = random_state
        self.n_in_ = None
        self.W_xh_ = None
        self.W_hh_ = None
        self.b_ = None
        self.X_ = None
        self.H_ = None
        self.dW_xh_ = None
        self.dW_hh_ = None
        self.db_ = None

    def _init_params(self, n_in):
        rng = np.random.default_rng(self.random_state)
        scale = np.sqrt(1.0 / max(n_in, 1))
        self.W_xh_ = rng.normal(scale=scale, size=(n_in, self.n_units))
        self.W_hh_ = rng.normal(scale=scale, size=(self.n_units, self.n_units))
        self.b_ = np.zeros(self.n_units, dtype=float)

    def forward(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 3:
            raise ValueError(f"Expected 3D input (N, T, D), got {X.shape}")
        N, T, D = X.shape
        if self.W_xh_ is None:
            self._init_params(D)
        self.X_ = X
        self.H_ = simple_rnn_forward(X, self.W_xh_, self.W_hh_, self.b_)
        h = self.H_[:, -1, :]
        if self.return_sequences:
            return self.H_
        return h

    def backward(self, grad):
        grad = np.asarray(grad, dtype=float)
        X = self.X_
        H = self.H_
        W_xh = self.W_xh_
        W_hh = self.W_hh_
        b = self.b_
        if X is None or H is None or W_xh is None or W_hh is None or b is None:
            raise RuntimeError("forward must be called before backward")
        N, T, _ = X.shape
        self.dW_xh_ = np.zeros_like(W_xh)
        self.dW_hh_ = np.zeros_like(W_hh)
        self.db_ = np.zeros_like(b)
        dX = np.zeros_like(X)
        dh = np.zeros((N, self.n_units), dtype=float)
        if grad.ndim == 2:
            dh = grad

        for t in range(T - 1, -1, -1):
            if grad.ndim == 3:
                dh = dh + grad[:, t, :]
            dtanh = dh * (1.0 - H[:, t, :] ** 2)
            self.dW_xh_ += X[:, t, :].T @ dtanh
            self.db_ += np.sum(dtanh, axis=0)
            h_prev = H[:, t - 1, :] if t > 0 else np.zeros((N, self.n_units))
            self.dW_hh_ += h_prev.T @ dtanh
            dX[:, t, :] = dtanh @ W_xh.T
            dh = dtanh @ W_hh.T
        return dX

    def step(self, lr=None, optimizer=None, key_prefix="rnn"):
        if self.dW_xh_ is None:
            raise RuntimeError("backward must be called before step")
        for name, param, grad in [("W_xh", self.W_xh_, self.dW_xh_),
                                   ("W_hh", self.W_hh_, self.dW_hh_),
                                   ("b", self.b_, self.db_)]:
            if optimizer is None:
                setattr(self, f"{name}_", param - lr * grad)
            else:
                setattr(self, f"{name}_", optimizer.update(param, grad, f"{key_prefix}.{name}"))


class LSTM(BaseLayer):
    """Long Short-Term Memory: f/i/g/o gates."""

    def __init__(self, n_units, return_sequences=False, random_state=None):
        self.n_units = int(n_units)
        self.return_sequences = bool(return_sequences)
        self.random_state = random_state
        self.n_in_ = None
        self.W_ = None
        self.U_ = None
        self.b_ = None
        self.X_ = None
        self.cache_ = None
        self.dW_ = None
        self.dU_ = None
        self.db_ = None

    def _init_params(self, n_in):
        rng = np.random.default_rng(self.random_state)
        scale = np.sqrt(1.0 / max(n_in, 1))
        self.W_ = rng.normal(scale=scale, size=(n_in, 4 * self.n_units))
        self.U_ = rng.normal(scale=scale, size=(self.n_units, 4 * self.n_units))
        self.b_ = np.zeros(4 * self.n_units, dtype=float)

    def forward(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 3:
            raise ValueError(f"Expected 3D input (N, T, D), got {X.shape}")
        N, T, D = X.shape
        if self.W_ is None:
            self._init_params(D)
        self.X_ = X
        u = self.n_units
        H, C, G = lstm_forward(X, self.W_, self.U_, self.b_, u)
        Hs = [H[:, t, :] for t in range(T)]
        Cs = [C[:, t, :] for t in range(T)]
        gates = [(G[:, t, :u], G[:, t, u:2 * u], G[:, t, 2 * u:3 * u], G[:, t, 3 * u:]) for t in range(T)]
        self.cache_ = (Hs, Cs, gates)
        h = H[:, -1, :]
        return H if self.return_sequences else h

    def backward(self, grad):
        grad = np.asarray(grad, dtype=float)
        X = self.X_
        W = self.W_
        U = self.U_
        b = self.b_
        cache = self.cache_
        if X is None or W is None or U is None or b is None or cache is None:
            raise RuntimeError("forward must be called before backward")
        N, T, _ = X.shape
        u = self.n_units
        Hs, Cs, gates = cache
        self.dW_ = np.zeros_like(W)
        self.dU_ = np.zeros_like(U)
        self.db_ = np.zeros_like(b)
        dX = np.zeros_like(X)
        dh, dc = np.zeros((N, u)), np.zeros((N, u))
        if grad.ndim == 2:
            dh = grad

        for t in range(T - 1, -1, -1):
            if grad.ndim == 3:
                dh = grad[:, t, :] + dh
            f, i, cc, o = gates[t]
            c_prev = Cs[t - 1] if t > 0 else np.zeros((N, u))
            do = dh * np.tanh(Cs[t]) * o * (1.0 - o)
            dc = dc + dh * o * (1.0 - np.tanh(Cs[t]) ** 2)
            df = dc * c_prev * f * (1.0 - f)
            di = dc * cc * i * (1.0 - i)
            dcc = dc * i * (1.0 - cc ** 2)
            dg = np.column_stack([df, di, dcc, do])
            self.dW_ += X[:, t, :].T @ dg
            h_prev = Hs[t - 1] if t > 0 else np.zeros((N, u))
            self.dU_ += h_prev.T @ dg
            self.db_ += np.sum(dg, axis=0)
            dX[:, t, :] = dg @ W.T
            dh = dg @ U.T
            dc = dc * f
        return dX

    def step(self, lr=None, optimizer=None, key_prefix="lstm"):
        if self.dW_ is None:
            raise RuntimeError("backward must be called before step")
        for name, param, grad in [("W", self.W_, self.dW_),
                                   ("U", self.U_, self.dU_),
                                   ("b", self.b_, self.db_)]:
            if optimizer is None:
                setattr(self, f"{name}_", param - lr * grad)
            else:
                setattr(self, f"{name}_", optimizer.update(param, grad, f"{key_prefix}.{name}"))


class GRU(BaseLayer):
    """Gated Recurrent Unit: z/r/h gates."""

    def __init__(self, n_units, return_sequences=False, random_state=None):
        self.n_units = int(n_units)
        self.return_sequences = bool(return_sequences)
        self.random_state = random_state
        self.n_in_ = None
        self.W_zr_ = None
        self.W_h_ = None
        self.U_zr_ = None
        self.U_h_ = None
        self.b_zr_ = None
        self.b_h_ = None
        self.X_ = None
        self.cache_ = None
        self.dW_zr_ = None
        self.dW_h_ = None
        self.dU_zr_ = None
        self.dU_h_ = None
        self.db_zr_ = None
        self.db_h_ = None

    def _init_params(self, n_in):
        rng = np.random.default_rng(self.random_state)
        scale = np.sqrt(1.0 / max(n_in, 1))
        u = self.n_units
        self.W_zr_ = rng.normal(scale=scale, size=(n_in, 2 * u))
        self.W_h_ = rng.normal(scale=scale, size=(n_in, u))
        self.U_zr_ = rng.normal(scale=scale, size=(u, 2 * u))
        self.U_h_ = rng.normal(scale=scale, size=(u, u))
        self.b_zr_ = np.zeros(2 * u, dtype=float)
        self.b_h_ = np.zeros(u, dtype=float)

    def forward(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 3:
            raise ValueError(f"Expected 3D input (N,T,D), got {X.shape}")
        N, T, D = X.shape
        if self.W_zr_ is None:
            self._init_params(D)
        self.X_ = X
        u = self.n_units
        H, G = gru_forward(X, self.W_zr_, self.W_h_, self.U_zr_, self.U_h_, self.b_zr_, self.b_h_, u)
        Hs = [H[:, t, :] for t in range(T)]
        gates = [(G[:, t, :u], G[:, t, u:2 * u], G[:, t, 2 * u:]) for t in range(T)]
        self.cache_ = (Hs, gates)
        h = H[:, -1, :]
        return H if self.return_sequences else h

    def backward(self, grad):
        grad = np.asarray(grad, dtype=float)
        X = self.X_
        W_zr = self.W_zr_
        W_h = self.W_h_
        U_zr = self.U_zr_
        U_h = self.U_h_
        b_zr = self.b_zr_
        b_h = self.b_h_
        cache = self.cache_
        if any(value is None for value in (X, W_zr, W_h, U_zr, U_h, b_zr, b_h, cache)):
            raise RuntimeError("forward must be called before backward")
        assert X is not None and W_zr is not None and W_h is not None
        assert U_zr is not None and U_h is not None and b_zr is not None and b_h is not None
        assert cache is not None
        N, T, _ = X.shape
        u = self.n_units
        Hs, gates = cache
        self.dW_zr_ = np.zeros_like(W_zr)
        self.dW_h_ = np.zeros_like(W_h)
        self.dU_zr_ = np.zeros_like(U_zr)
        self.dU_h_ = np.zeros_like(U_h)
        self.db_zr_ = np.zeros_like(b_zr)
        self.db_h_ = np.zeros_like(b_h)
        dX = np.zeros_like(X)
        dh = np.zeros((N, u))
        if grad.ndim == 2:
            dh = grad

        for t in range(T - 1, -1, -1):
            if grad.ndim == 3:
                dh = grad[:, t, :] + dh
            h_prev = Hs[t - 1] if t > 0 else np.zeros((N, u))
            z, r, h_tilde = gates[t]
            dh_tilde = dh * z * (1.0 - h_tilde ** 2)
            dz = dh * (h_tilde - h_prev) * z * (1.0 - z)
            dr = (dh_tilde @ U_h.T) * h_prev * r * (1.0 - r)

            dg_zr = np.column_stack([dz, dr])
            self.dW_zr_ += X[:, t, :].T @ dg_zr
            self.dU_zr_ += h_prev.T @ dg_zr
            self.db_zr_ += np.sum(dg_zr, axis=0)
            self.dW_h_ += X[:, t, :].T @ dh_tilde
            self.dU_h_ += (r * h_prev).T @ dh_tilde
            self.db_h_ += np.sum(dh_tilde, axis=0)
            dX[:, t, :] = dg_zr @ W_zr.T + dh_tilde @ W_h.T
            dh = dh * (1.0 - z) + dg_zr @ U_zr.T + dh_tilde @ U_h.T * r
        return dX

    def step(self, lr=None, optimizer=None, key_prefix="gru"):
        if self.dW_zr_ is None:
            raise RuntimeError("backward must be called before step")
        params = [
            ("W_zr", self.W_zr_, self.dW_zr_), ("W_h", self.W_h_, self.dW_h_),
            ("U_zr", self.U_zr_, self.dU_zr_), ("U_h", self.U_h_, self.dU_h_),
            ("b_zr", self.b_zr_, self.db_zr_), ("b_h", self.b_h_, self.db_h_),
        ]
        for name, param, grad in params:
            if optimizer is None:
                setattr(self, f"{name}_", param - lr * grad)
            else:
                setattr(self, f"{name}_", optimizer.update(param, grad, f"{key_prefix}.{name}"))
