from __future__ import annotations

import numpy as np


def stable_softmax(x, axis=-1):
    x = np.asarray(x, dtype=float)
    x = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def _broadcast_mask(mask, scores_shape):
    mask = np.asarray(mask)
    if mask.dtype == bool:
        return np.broadcast_to(mask, scores_shape), True
    return np.broadcast_to(mask.astype(float), scores_shape), False


def scaled_dot_product_attention(query, key, value, mask=None, causal=False, return_weights=False):
    query = np.asarray(query, dtype=float)
    key = np.asarray(key, dtype=float)
    value = np.asarray(value, dtype=float)

    if query.ndim < 2 or key.ndim < 2 or value.ndim < 2:
        raise ValueError("query, key, and value must have at least 2 dimensions")
    if query.shape[:-2] != key.shape[:-2] or query.shape[:-2] != value.shape[:-2]:
        raise ValueError("query, key, and value must share leading batch dimensions")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("query and key must have the same feature dimension")
    if key.shape[-2] != value.shape[-2]:
        raise ValueError("key and value must have the same sequence length")

    d_k = query.shape[-1]
    scores = np.matmul(query, np.swapaxes(key, -1, -2)) / np.sqrt(float(d_k))

    if causal:
        q_len = scores.shape[-2]
        k_len = scores.shape[-1]
        causal_mask = np.triu(np.ones((q_len, k_len), dtype=bool), k=1)
        scores = np.where(causal_mask, -1e9, scores)

    if mask is not None:
        mask, is_bool = _broadcast_mask(mask, scores.shape)
        if is_bool:
            scores = np.where(mask, scores, -1e9)
        else:
            scores = scores + mask

    weights = stable_softmax(scores, axis=-1)
    output = np.matmul(weights, value)
    if return_weights:
        return output, weights
    return output


def _split_heads(x, n_heads):
    x = np.asarray(x, dtype=float)
    if x.ndim != 3:
        raise ValueError("x must have shape (batch, steps, features)")
    n_heads = int(n_heads)
    if n_heads < 1:
        raise ValueError("n_heads must be at least 1")
    features = x.shape[-1]
    if features % n_heads != 0:
        raise ValueError("features must be divisible by n_heads")
    head_dim = features // n_heads
    batch, steps, _ = x.shape
    return x.reshape(batch, steps, n_heads, head_dim).transpose(0, 2, 1, 3)


def _merge_heads(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 4:
        raise ValueError("x must have shape (batch, n_heads, steps, head_dim)")
    batch, n_heads, steps, head_dim = x.shape
    return x.transpose(0, 2, 1, 3).reshape(batch, steps, n_heads * head_dim)


def multi_head_attention(
    x,
    w_q,
    w_k,
    w_v,
    w_o,
    b_q=None,
    b_k=None,
    b_v=None,
    b_o=None,
    n_heads=1,
    mask=None,
    causal=False,
    return_weights=False,
):
    x = np.asarray(x, dtype=float)
    w_q = np.asarray(w_q, dtype=float)
    w_k = np.asarray(w_k, dtype=float)
    w_v = np.asarray(w_v, dtype=float)
    w_o = np.asarray(w_o, dtype=float)

    if x.ndim != 3:
        raise ValueError("x must have shape (batch, steps, features)")
    if x.shape[-1] != w_q.shape[0] or w_q.shape != w_k.shape or w_q.shape != w_v.shape:
        raise ValueError("projection matrices must all be square and match the input feature dimension")
    if w_o.shape != w_q.shape:
        raise ValueError("w_o must match the projection matrix shape")

    b_q = np.zeros(w_q.shape[1], dtype=float) if b_q is None else np.asarray(b_q, dtype=float)
    b_k = np.zeros(w_q.shape[1], dtype=float) if b_k is None else np.asarray(b_k, dtype=float)
    b_v = np.zeros(w_q.shape[1], dtype=float) if b_v is None else np.asarray(b_v, dtype=float)
    b_o = np.zeros(w_q.shape[1], dtype=float) if b_o is None else np.asarray(b_o, dtype=float)

    q = x @ w_q + b_q
    k = x @ w_k + b_k
    v = x @ w_v + b_v

    qh = _split_heads(q, n_heads)
    kh = _split_heads(k, n_heads)
    vh = _split_heads(v, n_heads)

    attn_out, weights = scaled_dot_product_attention(qh, kh, vh, mask=mask, causal=causal, return_weights=True)
    merged = _merge_heads(attn_out)
    output = merged @ w_o + b_o
    if return_weights:
        return output, weights
    return output


def sinusoidal_positional_encoding(length, d_model, base=10000.0):
    length = int(length)
    d_model = int(d_model)
    if length < 0:
        raise ValueError("length must be non-negative")
    if d_model < 1:
        raise ValueError("d_model must be at least 1")
    positions = np.arange(length, dtype=float)[:, None]
    dims = np.arange(d_model, dtype=float)[None, :]
    angle_rates = 1.0 / np.power(base, (2 * np.floor(dims / 2.0)) / max(d_model, 1))
    angles = positions * angle_rates
    encoding = np.zeros((length, d_model), dtype=float)
    encoding[:, 0::2] = np.sin(angles[:, 0::2])
    if d_model > 1:
        encoding[:, 1::2] = np.cos(angles[:, 1::2])
    return encoding


__all__ = [
    "multi_head_attention",
    "scaled_dot_product_attention",
    "sinusoidal_positional_encoding",
    "stable_softmax",
]
