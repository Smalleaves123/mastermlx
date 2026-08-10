import importlib
import threading

import numpy as np
import pytest

from mastermlx.base.checkpoint import _version_parts
from mastermlx.clustering import MiniBatchKMeans
from mastermlx.data.model_selection import _split_cv
from mastermlx.ensemble import (
    StackingClassifier,
    StackingRegressor,
    VotingClassifier,
    VotingRegressor,
)
from mastermlx.linear_models import (
    LinearRegression,
    LogisticRegression,
    RANSACRegressor,
    SGDClassifier,
)
from mastermlx.math_tools.lr_find import lr_find
from mastermlx.neural_net import Adam, Dense, MLPRegressor, Sequential
from mastermlx.preprocessing import QuantileTransform
from mastermlx.selection import RFE, SelectFromModel, f_classif
from mastermlx.tabular import TabularExperiment


def test_f_classif_supports_more_than_344_samples():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(345, 3))
    y = np.arange(X.shape[0]) % 2

    scores, pvalues = f_classif(X, y)

    assert np.all(np.isfinite(scores))
    assert np.all(np.isfinite(pvalues))
    assert np.all((pvalues >= 0.0) & (pvalues <= 1.0))


def test_quantile_transform_centers_constant_and_tied_values():
    constant = np.ones((5, 1))
    uniform = QuantileTransform(output_distribution="uniform").fit_transform(constant)
    normal = QuantileTransform(output_distribution="normal").fit_transform(constant)

    assert np.allclose(uniform, 0.5)
    assert np.allclose(normal, 0.0, atol=1e-6)

    discrete = np.array([[0.0], [1.0], [1.0], [1.0], [2.0]])
    transformed = QuantileTransform(output_distribution="uniform").fit_transform(discrete)
    assert np.allclose(transformed[1:4], 0.5)
    assert transformed[0, 0] == 0.0
    assert transformed[-1, 0] == 1.0


def test_sequential_load_migrates_legacy_optimizer_keys(tmp_path):
    model = Sequential([Dense(2, 2, random_state=0)], optimizer="adam")
    model.optimizer_ = Adam()
    model.optimizer_._m = {"dense0.W": np.full((2, 2), 3.0)}
    model.optimizer_._v = {"dense0.W": np.full((2, 2), 4.0)}
    checkpoint = tmp_path / "legacy-sequential.mlx"
    model.save_checkpoint(checkpoint)

    with pytest.warns(UserWarning, match="migrated legacy Sequential optimizer"):
        restored = Sequential.load_checkpoint(checkpoint)

    assert "dense0.W" not in restored.optimizer_._m
    assert np.all(restored.optimizer_._m["layer0.dense.W"] == 3.0)
    assert np.all(restored.optimizer_._v["layer0.dense.W"] == 4.0)


def test_scheduler_supports_keyword_only_and_opaque_steps():
    class KeywordScheduler:
        def step(self, *, metric):
            self.metric = metric

    keyword = KeywordScheduler()
    Sequential([], lr_scheduler=keyword)._on_epoch_end(1, {"monitor_loss": 1.25})
    assert keyword.metric == 1.25

    class OpaqueStep:
        @property
        def __signature__(self):
            raise ValueError("signature unavailable")

        def __call__(self, metric):
            self.metric = metric

    class OpaqueScheduler:
        step = OpaqueStep()

    opaque = OpaqueScheduler()
    Sequential([], lr_scheduler=opaque)._on_epoch_end(1, {"monitor_loss": 2.5})
    assert opaque.step.metric == 2.5


def test_opaque_splitter_without_groups_parameter_remains_usable():
    class OpaqueSplit:
        @property
        def __signature__(self):
            raise ValueError("signature unavailable")

        def __call__(self, X, y=None):
            return [(np.array([0, 1]), np.array([2, 3]))]

    class Splitter:
        split = OpaqueSplit()

    splits = list(
        _split_cv(
            Splitter(),
            np.arange(8.0).reshape(4, 2),
            np.arange(4),
            groups=np.arange(4),
        )
    )
    assert len(splits) == 1


def test_lr_find_uses_a_working_copy_for_slots_and_unpickleable_state():
    class SlotsModel:
        __slots__ = ("lr", "loss_")

        def __init__(self):
            self.lr = 0.1
            self.loss_ = []

        def fit(self, X, y):
            self.loss_.append(float(self.lr))
            return self

    slots_model = SlotsModel()
    lrs, losses = lr_find(slots_model, [[0.0], [1.0]], [0.0, 1.0], n_iters=2)
    assert lrs.shape == losses.shape == (2,)
    assert slots_model.lr == 0.1
    assert slots_model.loss_ == []

    class LockedModel:
        def __init__(self):
            self.lr = 0.1
            self.loss_ = []
            self._lock = threading.Lock()

        def fit(self, X, y):
            self.loss_.append(float(self.lr))
            return self

        def score(self, X, y):
            return 0.0

    locked_model = LockedModel()
    lr_find(locked_model, [[0.0], [1.0]], [0.0, 1.0], n_iters=1)
    assert locked_model.loss_ == []


def test_multiclass_linear_importance_uses_feature_axis():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(45, 5))
    y = np.arange(X.shape[0]) % 3

    selected = SelectFromModel(
        LogisticRegression(n_iter=20, random_state=0),
        threshold="mean",
    ).fit(X, y)
    recursive = RFE(
        LogisticRegression(n_iter=20, random_state=0),
        n_features_to_select=2,
        step=0.4,
    ).fit(X, y)

    assert selected.support_.shape == (X.shape[1],)
    assert recursive.support_.shape == (X.shape[1],)
    assert np.sum(recursive.support_) == 2


def test_select_from_model_clones_the_user_estimator():
    X = np.arange(12.0).reshape(6, 2)
    y = np.array([0, 0, 0, 1, 1, 1])
    estimator = LogisticRegression(n_iter=10, random_state=0)

    selector = SelectFromModel(estimator).fit(X, y)

    assert estimator.coef_ is None
    assert selector.estimator_ is not estimator


def test_pep440_prerelease_versions_keep_major_compatibility_gate():
    assert _version_parts("0.2.0rc1") == (0, 2, 0)
    assert _version_parts("1!2.0.0rc1+cpu") == (2, 0, 0)


def test_tabular_default_cross_validation_remains_reproducible():
    X = np.linspace(-2.0, 2.0, 30).reshape(-1, 1)
    y = X[:, 0] ** 3 + 0.2 * X[:, 0]
    experiment = TabularExperiment(
        LinearRegression(),
        search=None,
        task="regression",
    ).fit(X, y)

    first = experiment.cv_score(X, y)
    second = experiment.cv_score(X, y)

    assert np.array_equal(first, second)


def test_training_exposes_best_loss_without_mislabeling_training_as_validation():
    X = np.arange(8.0).reshape(-1, 1)
    y = 2.0 * X[:, 0] + 1.0
    model = MLPRegressor(
        hidden_layer_sizes=(),
        n_iter=2,
        tol=0.0,
        random_state=0,
        validation_split=0.0,
    ).fit(X, y)

    assert np.isfinite(model.best_loss_)
    assert model.best_val_loss_ is None


def test_corrected_contracts_reject_invalid_sgd_loss_and_use_floor_rfe_step():
    with pytest.raises(ValueError, match="loss must be one of"):
        SGDClassifier(loss="hingee")
    assert RFE(LinearRegression(), step=0.34)._step(5) == 1


def test_minibatch_kmeans_uses_summed_inertia_and_rejects_too_many_clusters():
    X = np.array([[0.0], [1.0], [9.0], [10.0]])
    model = MiniBatchKMeans(
        n_clusters=2,
        batch_size=4,
        max_iter=5,
        n_init=1,
        random_state=0,
    ).fit(X)
    expected = np.sum((X - model.cluster_centers_[model.labels_]) ** 2)

    assert np.isclose(model.inertia_, expected)
    with pytest.raises(ValueError, match="n_clusters"):
        MiniBatchKMeans(n_clusters=5).fit(X)


@pytest.mark.parametrize(
    "ensemble",
    [
        VotingClassifier([LogisticRegression(n_iter=10, random_state=0)]),
        StackingClassifier(
            [LogisticRegression(n_iter=10, random_state=0)],
            cv=2,
            random_state=0,
        ),
    ],
)
def test_classifier_ensembles_clone_user_estimators(ensemble):
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    original = ensemble.estimators[0]

    ensemble.fit(X, y)

    assert original.coef_ is None
    assert ensemble.estimators_[0] is not original


@pytest.mark.parametrize(
    "ensemble",
    [
        VotingRegressor([LinearRegression()]),
        StackingRegressor([LinearRegression()], cv=2, random_state=0),
    ],
)
def test_regressor_ensembles_clone_user_estimators(ensemble):
    X = np.arange(6.0).reshape(-1, 1)
    y = 2.0 * X[:, 0] + 1.0
    original = ensemble.estimators[0]

    ensemble.fit(X, y)

    assert original.coef_ is None
    assert ensemble.estimators_[0] is not original


def test_ransac_does_not_replace_explicit_zero_parameters():
    X = np.arange(6.0).reshape(-1, 1)
    y = 2.0 * X[:, 0] + 1.0

    with pytest.raises(ValueError, match="min_samples"):
        RANSACRegressor(min_samples=0).fit(X, y)
    with pytest.raises(ValueError, match="residual_threshold"):
        RANSACRegressor(residual_threshold=-1.0).fit(X, y)

    exact = RANSACRegressor(
        min_samples=2,
        residual_threshold=0.0,
        max_trials=10,
        random_state=0,
    ).fit(X, y)
    assert exact.inlier_mask_.any()


def test_invalid_backend_environment_value_warns_before_falling_back(monkeypatch):
    import mastermlx.config as config

    monkeypatch.setenv("MASTERML_BACKEND", "invalid-backend")
    try:
        with pytest.warns(RuntimeWarning, match="falling back to 'auto'"):
            importlib.reload(config)
        assert config.get_backend() == "auto"
    finally:
        monkeypatch.delenv("MASTERML_BACKEND", raising=False)
        importlib.reload(config)


def test_hard_voting_honors_weights():
    class ConstantClassifier:
        def __init__(self, label):
            self.label = label

        def fit(self, X, y):
            self.classes_ = np.unique(y)
            return self

        def predict(self, X):
            return np.full(np.asarray(X).shape[0], self.label)

    X = np.arange(4.0).reshape(-1, 1)
    y = np.array([0, 0, 1, 1])
    model = VotingClassifier(
        [ConstantClassifier(0), ConstantClassifier(1)],
        weights=[0.1, 0.9],
        voting="hard",
    ).fit(X, y)

    assert np.array_equal(model.predict(X), np.ones(X.shape[0], dtype=int))
