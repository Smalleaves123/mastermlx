import numpy as np

from mastermlx import LinearRegression, SGDClassifier, SGDRegressor
from mastermlx.data import OnlineTabularExperiment


def test_sgd_estimators_support_incremental_updates():
    X = np.arange(40, dtype=float).reshape(-1, 1)
    y_reg = 2.0 * X[:, 0] + 1.0
    regressor = SGDRegressor(eta0=0.001, max_iter=1, random_state=0)
    for start in range(0, X.shape[0], 5):
        regressor.partial_fit(X[start : start + 5], y_reg[start : start + 5])

    y_cls = (X[:, 0] >= 20).astype(float)
    classifier = SGDClassifier(loss="log_loss", eta0=0.01, max_iter=1, random_state=0)
    for start in range(0, X.shape[0], 5):
        classifier.partial_fit(
            X[start : start + 5],
            y_cls[start : start + 5],
            classes=[0, 1] if start == 0 else None,
        )

    assert np.isfinite(regressor.predict([[40.0]])).all()
    assert classifier.classes_.tolist() == [0.0, 1.0]
    assert classifier.predict([[35.0]]).shape == (1,)


def test_online_experiment_tracks_batches_drift_and_sliding_window():
    model = SGDRegressor(eta0=0.001, max_iter=1, random_state=0)
    online = OnlineTabularExperiment(model, task="regression", drift_thresholds={"psi": 0.1})
    first = online.partial_fit(np.arange(4, dtype=float).reshape(-1, 1), [1.0, 3.0, 5.0, 7.0])
    second = online.partial_fit(
        np.arange(100, 104, dtype=float).reshape(-1, 1),
        [201.0, 203.0, 205.0, 207.0],
        batch_id="shift",
    )

    assert first["metrics"] is None
    assert second["metrics"] is not None
    assert second["drift"]["detected"]
    assert online.report()["n_updates"] == 2

    windowed = OnlineTabularExperiment(LinearRegression(), task="regression", window_size=3)
    windowed.partial_fit([[0.0], [1.0], [2.0]], [0.0, 1.0, 2.0])
    windowed.partial_fit([[3.0], [4.0]], [3.0, 4.0])
    state = windowed.report()
    assert state["mode"] == "sliding_window"
    assert state["window_size"] == 3
    assert state["n_samples"] == 5


def test_online_experiment_supports_delayed_labels():
    model = SGDClassifier(loss="log_loss", eta0=0.02, max_iter=1, random_state=0)
    online = OnlineTabularExperiment(model, task="classification")
    online.partial_fit([[0.0], [1.0], [2.0], [3.0]], [0, 0, 1, 1], classes=[0, 1])

    pending = online.predict_unlabeled([[0.5], [2.5]], sample_ids=["a", "b"])
    labeled = online.update_labels(pending["sample_ids"], [0, 1])

    assert labeled["n_samples"] == 2
    assert "accuracy" in labeled["metrics"]
    assert online.report()["pending_labels"] == 0
