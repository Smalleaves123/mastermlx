from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace

try:  # Optional dependency.
    import seaborn as sns
except ModuleNotFoundError:  # pragma: no cover - fallback path
    sns = None

try:  # Optional dependency.
    import plotly.express as px
except ModuleNotFoundError:  # pragma: no cover - fallback path
    px = None

from ..utils.validation import check_1d_array, check_2d_array


def _prepare_ax(ax=None, figsize=(6, 4)):
    if ax is not None:
        return ax, ax.figure
    fig, ax = plt.subplots(figsize=figsize)
    return ax, fig


def plot_loss_curve(losses, ax=None, title="Training loss"):
    losses = np.asarray(losses, dtype=float)
    if losses.ndim != 1 or losses.size == 0:
        raise ValueError("losses must be a non-empty 1D sequence")
    ax, fig = _prepare_ax(ax)
    x = np.arange(1, losses.size + 1)
    if sns is not None:
        sns.lineplot(x=x, y=losses, ax=ax, marker="o")
    else:
        ax.plot(x, losses, marker="o")
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return ax


def plot_confusion_matrix(y_true, y_pred, labels=None, normalize=False, ax=None, title="Confusion matrix"):
    y_true = check_1d_array(y_true, name="y_true")
    y_pred = check_1d_array(y_pred, name="y_pred")
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")

    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    labels = np.asarray(labels)
    index = {label: i for i, label in enumerate(labels)}
    cm = np.zeros((labels.size, labels.size), dtype=float)
    for yt, yp in zip(y_true, y_pred):
        cm[index[yt], index[yp]] += 1
    if normalize:
        row_sum = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sum, out=np.zeros_like(cm), where=row_sum != 0)

    ax, fig = _prepare_ax(ax)
    if sns is not None:
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f" if normalize else ".0f",
            cmap="Blues",
            cbar=False,
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
        )
    else:
        im = ax.imshow(cm, cmap="Blues")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(np.arange(labels.size), labels=labels)
        ax.set_yticks(np.arange(labels.size), labels=labels)
        for i in range(labels.size):
            for j in range(labels.size):
                value = cm[i, j]
                ax.text(j, i, f"{value:.2f}" if normalize else f"{value:.0f}", ha="center", va="center")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    return ax


def plot_scatter_2d(X, y=None, ax=None, title="2D scatter", palette="deep"):
    X = check_2d_array(X)
    if X.shape[1] != 2:
        raise ValueError("X must be 2D with exactly 2 features")
    ax, fig = _prepare_ax(ax, figsize=(6, 5))
    if y is None:
        ax.scatter(X[:, 0], X[:, 1], s=35, alpha=0.85)
    else:
        y = check_1d_array(y, name="y")
        if y.shape[0] != X.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        if sns is not None:
            sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette=palette, ax=ax, s=40, edgecolor="none")
            ax.legend(title="Class", loc="best")
        else:
            labels = np.unique(y)
            cmap = plt.get_cmap("tab10")
            for idx, label in enumerate(labels):
                mask = y == label
                ax.scatter(X[mask, 0], X[mask, 1], s=40, alpha=0.85, color=cmap(idx), label=str(label))
            ax.legend(title="Class", loc="best")
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    fig.tight_layout()
    return ax


def plot_explained_variance(explained_variance_ratio, ax=None, title="Explained variance"):
    values = np.asarray(explained_variance_ratio, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("explained_variance_ratio must be a non-empty 1D sequence")
    ax, fig = _prepare_ax(ax)
    x = np.arange(1, values.size + 1)
    if sns is not None:
        sns.barplot(x=x, y=values, ax=ax, color="#4C72B0")
    else:
        ax.bar(x, values, color="#4C72B0")
    ax.plot(x, np.cumsum(values), color="#DD8452", marker="o", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Component")
    ax.set_ylabel("Ratio")
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_ylim(0.0, 1.05)
    fig.tight_layout()
    return ax


def plot_interactive_scatter(X, y=None, title="Interactive scatter"):
    X = check_2d_array(X)
    if X.shape[1] != 2:
        raise ValueError("X must be 2D with exactly 2 features")
    data = {"x1": X[:, 0], "x2": X[:, 1]}
    if y is not None:
        y = check_1d_array(y, name="y")
        if y.shape[0] != X.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        data["label"] = y.astype(str)
        if px is not None:
            return px.scatter(data, x="x1", y="x2", color="label", title=title)
        return SimpleNamespace(data=[data], layout={"title": title})
    if px is not None:
        return px.scatter(data, x="x1", y="x2", title=title)
    return SimpleNamespace(data=[data], layout={"title": title})


plot_cm = plot_confusion_matrix
plot_var = plot_explained_variance
plot_scatter = plot_scatter_2d
plot_loss = plot_loss_curve
plot_int_scatter = plot_interactive_scatter
