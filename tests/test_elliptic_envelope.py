import numpy as np

from mastermlx.anomaly import EllipticEnvelope


def test_elliptic_envelope_flags_far_outlier():
    rng = np.random.default_rng(0)
    X_core = rng.normal(loc=0.0, scale=0.2, size=(40, 2))
    X = np.vstack([X_core, np.array([[4.0, 4.0]])])

    model = EllipticEnvelope(contamination=0.1, max_iter=20).fit(X)
    pred = model.predict(X)

    assert pred.shape == (41,)
    assert pred[-1] == -1


def test_elliptic_envelope_scores_outlier_higher():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 2))
    model = EllipticEnvelope(contamination=0.1).fit(X)

    scores = model.score_samples([[0.0, 0.0], [5.0, 5.0]])

    assert scores.shape == (2,)
    assert scores[1] > scores[0]


def test_elliptic_envelope_support_and_covariance_shapes():
    X = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [0.1, 0.1],
            [2.0, 2.0],
        ]
    )

    model = EllipticEnvelope(contamination=0.2).fit(X)

    assert model.location_.shape == (2,)
    assert model.covariance_.shape == (2, 2)
    assert model.support_.shape == (5,)
