from __future__ import annotations

import numpy as np


def plot_chain(points, ax=None, *, show_axes=True, annotate=False):
    """Plot a 2D or 3D kinematic chain.

    `points` must be an array of shape (N, 2) or (N, 3).
    """

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] not in {2, 3}:
        raise ValueError("points must have shape (N, 2) or (N, 3)")

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency failure path
        raise RuntimeError("matplotlib is required for plotting") from exc

    if ax is None:
        if pts.shape[1] == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
        else:
            _, ax = plt.subplots()

    if pts.shape[1] == 3:
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], marker="o")
        if annotate:
            for i, p in enumerate(pts):
                ax.text(p[0], p[1], p[2], str(i))
    else:
        ax.plot(pts[:, 0], pts[:, 1], marker="o")
        if annotate:
            for i, p in enumerate(pts):
                ax.text(p[0], p[1], str(i))

    if show_axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        if pts.shape[1] == 3:
            ax.set_zlabel("z")
    return ax
