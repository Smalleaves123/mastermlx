import matplotlib

matplotlib.use("Agg")

import numpy as np

from mastermlx.viz import (
    plot_confusion_matrix,
    plot_explained_variance,
    plot_interactive_scatter,
    plot_loss_curve,
    plot_scatter_2d,
)


def test_viz_loss_and_confusion_matrix_smoke():
    losses = [1.0, 0.6, 0.3]
    ax = plot_loss_curve(losses)
    assert ax.get_title() == "Training loss"

    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    ax = plot_confusion_matrix(y_true, y_pred)
    assert ax.get_title() == "Confusion matrix"


def test_viz_scatter_and_variance_smoke():
    X = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 1.5],
    ])
    y = np.array([0, 1, 1])

    ax = plot_scatter_2d(X, y)
    assert ax.get_title() == "2D scatter"

    ax = plot_explained_variance([0.7, 0.2, 0.1])
    assert ax.get_title() == "Explained variance"


def test_viz_plotly_scatter_returns_figure():
    X = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 1.5],
    ])
    fig = plot_interactive_scatter(X, np.array([0, 1, 1]))
    assert fig.data
