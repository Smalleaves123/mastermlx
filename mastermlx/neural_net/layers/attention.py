from __future__ import annotations

import numpy as np

from ...base import BaseLayer


def _stable_softmax(x, axis=-1):
    x = np.asarray(x, dtype=float)
    x = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=axis, keepdims=True)


class AttentionPooling1D(BaseLayer):
    """Learned attention pooling over a sequence axis.

    Input shape: (batch, steps, features)
    Output shape: (batch, features)
    """

    def __init__(self, hidden_size=64, random_state=None):
        self.hidden_size = int(hidden_size)
        self.random_state = random_state
        self.input_dim_ = None
        self.W_ = None
        self.b_ = None
        self.u_ = None
        self.c_ = None
        self.X_ = None
        self.H_ = None
        self.alpha_ = None
        self.dW_ = None
        self.db_ = None
        self.du_ = None
        self.dc_ = None

    def _init_params(self, input_dim):
        rng = np.random.default_rng(self.random_state)
        scale = np.sqrt(1.0 / max(int(input_dim), 1))
        self.W_ = rng.normal(scale=scale, size=(int(input_dim), self.hidden_size))
        self.b_ = np.zeros(self.hidden_size, dtype=float)
        self.u_ = rng.normal(scale=scale, size=(self.hidden_size,))
        self.c_ = 0.0

    def forward(self, X):
        X = np.asarray(X, dtype=float)
        if X.size == 0:
            raise ValueError("Expected a non-empty array")
        if X.ndim != 3:
            raise ValueError(f"Expected 3D array, got shape {X.shape}")
        if self.W_ is None:
            self.input_dim_ = X.shape[2]
            self._init_params(self.input_dim_)
        if X.shape[2] != self.input_dim_:
            raise ValueError(f"Expected {self.input_dim_} features, got {X.shape[2]}")

        self.X_ = X
        self.H_ = np.tanh(X @ self.W_ + self.b_)
        scores = self.H_ @ self.u_ + self.c_
        self.alpha_ = _stable_softmax(scores, axis=1)
        return np.sum(self.alpha_[..., None] * X, axis=1)

    def backward(self, grad):
        grad = np.asarray(grad, dtype=float)
        if self.X_ is None or self.H_ is None or self.alpha_ is None:
            raise RuntimeError("forward must be called before backward")
        if grad.ndim != 2 or grad.shape != (self.X_.shape[0], self.X_.shape[2]):
            raise ValueError("Gradient shape does not match pooled output shape")

        direct = self.alpha_[..., None] * grad[:, None, :]
        d_alpha = np.sum(grad[:, None, :] * self.X_, axis=2)
        d_scores = self.alpha_ * (d_alpha - np.sum(d_alpha * self.alpha_, axis=1, keepdims=True))

        self.du_ = np.sum(self.H_ * d_scores[..., None], axis=(0, 1))
        self.dc_ = float(np.sum(d_scores))

        d_hidden = d_scores[..., None] * self.u_[None, None, :]
        d_pre = d_hidden * (1.0 - self.H_ ** 2)
        self.dW_ = np.einsum("btd,bth->dh", self.X_, d_pre)
        self.db_ = np.sum(d_pre, axis=(0, 1))

        dX = direct + d_pre @ self.W_.T
        return dX

    def step(self, lr=None, optimizer=None, key_prefix="attn_pool"):
        if self.dW_ is None or self.db_ is None or self.du_ is None or self.dc_ is None:
            raise RuntimeError("backward must be called before step")
        if optimizer is None:
            if lr is None:
                raise ValueError("lr must be provided when optimizer is None")
            self.W_ -= lr * self.dW_
            self.b_ -= lr * self.db_
            self.u_ -= lr * self.du_
            self.c_ -= lr * self.dc_
            return
        self.W_ = optimizer.update(self.W_, self.dW_, f"{key_prefix}.W")
        self.b_ = optimizer.update(self.b_, self.db_, f"{key_prefix}.b")
        self.u_ = optimizer.update(self.u_, self.du_, f"{key_prefix}.u")
        self.c_ = optimizer.update(np.asarray(self.c_, dtype=float), np.asarray(self.dc_, dtype=float), f"{key_prefix}.c").item()


class MultiHeadAttention(BaseLayer):
    """Self-attention block with learned Q/K/V projections.

    Input and output shapes are both (batch, steps, features).
    """

    def __init__(self, n_features, n_heads=1, random_state=None, causal=False):
        self.n_features = int(n_features)
        self.n_heads = int(n_heads)
        self.random_state = random_state
        self.causal = bool(causal)
        if self.n_heads < 1:
            raise ValueError("n_heads must be at least 1")
        if self.n_features % self.n_heads != 0:
            raise ValueError("n_features must be divisible by n_heads")
        self.head_dim_ = self.n_features // self.n_heads
        self.W_q_ = None
        self.W_k_ = None
        self.W_v_ = None
        self.W_o_ = None
        self.b_q_ = None
        self.b_k_ = None
        self.b_v_ = None
        self.b_o_ = None
        self.X_ = None
        self.Q_ = None
        self.K_ = None
        self.V_ = None
        self.A_ = None
        self.H_ = None
        self.dW_q_ = None
        self.dW_k_ = None
        self.dW_v_ = None
        self.dW_o_ = None
        self.db_q_ = None
        self.db_k_ = None
        self.db_v_ = None
        self.db_o_ = None

    def _init_params(self):
        rng = np.random.default_rng(self.random_state)
        scale = np.sqrt(1.0 / max(self.n_features, 1))
        shape = (self.n_features, self.n_features)
        self.W_q_ = rng.normal(scale=scale, size=shape)
        self.W_k_ = rng.normal(scale=scale, size=shape)
        self.W_v_ = rng.normal(scale=scale, size=shape)
        self.W_o_ = rng.normal(scale=scale, size=shape)
        self.b_q_ = np.zeros(self.n_features, dtype=float)
        self.b_k_ = np.zeros(self.n_features, dtype=float)
        self.b_v_ = np.zeros(self.n_features, dtype=float)
        self.b_o_ = np.zeros(self.n_features, dtype=float)

    def _reshape_heads(self, X):
        b, t, _ = X.shape
        return X.reshape(b, t, self.n_heads, self.head_dim_).transpose(0, 2, 1, 3)

    def _merge_heads(self, X):
        b, h, t, d = X.shape
        return X.transpose(0, 2, 1, 3).reshape(b, t, h * d)

    def forward(self, X):
        X = np.asarray(X, dtype=float)
        if X.size == 0:
            raise ValueError("Expected a non-empty array")
        if X.ndim != 3:
            raise ValueError(f"Expected 3D array, got shape {X.shape}")
        if X.shape[2] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {X.shape[2]}")
        if self.W_q_ is None:
            self._init_params()

        self.X_ = X
        self.Q_ = X @ self.W_q_ + self.b_q_
        self.K_ = X @ self.W_k_ + self.b_k_
        self.V_ = X @ self.W_v_ + self.b_v_

        Qh = self._reshape_heads(self.Q_)
        Kh = self._reshape_heads(self.K_)
        Vh = self._reshape_heads(self.V_)

        scores = np.einsum("bhid,bhjd->bhij", Qh, Kh) / np.sqrt(self.head_dim_)
        if self.causal:
            t = scores.shape[-1]
            mask = np.triu(np.ones((t, t), dtype=bool), k=1)
            scores = np.where(mask[None, None, :, :], -1e9, scores)

        self.A_ = _stable_softmax(scores, axis=-1)
        self.H_ = np.einsum("bhij,bhjd->bhid", self.A_, Vh)
        out = self._merge_heads(self.H_)
        return out @ self.W_o_ + self.b_o_

    def backward(self, grad):
        grad = np.asarray(grad, dtype=float)
        if self.X_ is None or self.A_ is None or self.H_ is None:
            raise RuntimeError("forward must be called before backward")
        if grad.shape != (self.X_.shape[0], self.X_.shape[1], self.n_features):
            raise ValueError("Gradient shape does not match attention output shape")

        out = self._merge_heads(self.H_)
        self.dW_o_ = np.einsum("btd,bto->do", out, grad)
        self.db_o_ = np.sum(grad, axis=(0, 1))

        d_out = grad @ self.W_o_.T
        dH = self._reshape_heads(d_out)

        Vh = self._reshape_heads(self.V_)
        Qh = self._reshape_heads(self.Q_)
        Kh = self._reshape_heads(self.K_)

        dA = np.einsum("bhid,bhjd->bhij", dH, Vh)
        dVh = np.einsum("bhij,bhid->bhjd", self.A_, dH)

        dot = np.sum(dA * self.A_, axis=-1, keepdims=True)
        dScores = self.A_ * (dA - dot)

        scale = 1.0 / np.sqrt(self.head_dim_)
        dQh = np.einsum("bhij,bhjd->bhid", dScores, Kh) * scale
        dKh = np.einsum("bhij,bhid->bhjd", dScores, Qh) * scale

        dQ = self._merge_heads(dQh)
        dK = self._merge_heads(dKh)
        dV = self._merge_heads(dVh)

        X_flat = self.X_.reshape(-1, self.n_features)
        dQ_flat = dQ.reshape(-1, self.n_features)
        dK_flat = dK.reshape(-1, self.n_features)
        dV_flat = dV.reshape(-1, self.n_features)

        self.dW_q_ = X_flat.T @ dQ_flat
        self.dW_k_ = X_flat.T @ dK_flat
        self.dW_v_ = X_flat.T @ dV_flat
        self.db_q_ = np.sum(dQ_flat, axis=0)
        self.db_k_ = np.sum(dK_flat, axis=0)
        self.db_v_ = np.sum(dV_flat, axis=0)

        dX = dQ @ self.W_q_.T + dK @ self.W_k_.T + dV @ self.W_v_.T
        return dX

    def step(self, lr=None, optimizer=None, key_prefix="mha"):
        params = [
            ("W_q", "dW_q"),
            ("W_k", "dW_k"),
            ("W_v", "dW_v"),
            ("W_o", "dW_o"),
            ("b_q", "db_q"),
            ("b_k", "db_k"),
            ("b_v", "db_v"),
            ("b_o", "db_o"),
        ]
        for param_name, grad_name in params:
            param = getattr(self, f"{param_name}_")
            grad = getattr(self, f"{grad_name}_")
            if grad is None:
                raise RuntimeError("backward must be called before step")
            if optimizer is None:
                if lr is None:
                    raise ValueError("lr must be provided when optimizer is None")
                updated = param - lr * grad
            else:
                updated = optimizer.update(param, grad, f"{key_prefix}.{param_name}")
            if param_name == "c":
                setattr(self, f"{param_name}_", float(updated))
            else:
                setattr(self, f"{param_name}_", updated)
