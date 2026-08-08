import numpy as np
import pytest

from mastermlx.base import BaseTransformer
from mastermlx.clustering import KMeans, MiniBatchKMeans
from mastermlx.ensemble import CalibratedClassifierCV, MultiOutputClassifier
from mastermlx.linear_models import LogisticRegression, Perceptron, SGDClassifier
from mastermlx.neighbors import KNNClassifier, KNNRegressor
from mastermlx.neural_net import Adam, AdaGrad, AdamW, OptimizerConfig, Sequential, build_optimizer
from mastermlx.preprocessing import Binarizer, LabelEncoder
from mastermlx.selection import SelectFromModel, SelectKBest, VarianceThreshold, f_classif, f_regression
from mastermlx.svm import LinearSVR, SVC
from mastermlx.trees import DecisionTreeClassifier, DecisionTreeRegressor
from mastermlx.utils import clone, r2_score


X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
Y_CLASS = np.array([0, 0, 1, 1])
Y_REG = np.array([0.0, 1.0, 1.0, 2.0])


@pytest.mark.parametrize(
    "estimator",
    [
        KNNClassifier(k=1),
        DecisionTreeClassifier(max_depth=2),
        SVC(kernel="linear", max_iter=100, random_state=0),
    ],
)
def test_classifier_predict_preserves_single_sample_axis(estimator):
    estimator.fit(X, Y_CLASS)

    assert estimator.predict(X[:1]).shape == (1,)


@pytest.mark.parametrize(
    "estimator",
    [KNNRegressor(k=2), DecisionTreeRegressor(max_depth=2), LinearSVR(max_iter=200)],
)
def test_regressor_predict_preserves_single_sample_axis(estimator):
    estimator.fit(X, Y_REG)

    assert estimator.predict(X[:1]).shape == (1,)


def test_knn_regressor_score_uses_r2():
    model = KNNRegressor(k=2).fit(X, Y_REG)

    assert np.isclose(model.score(X, Y_REG), r2_score(Y_REG, model.predict(X)))


def test_cluster_predict_preserves_single_sample_axis_and_inertia_is_sum():
    full = KMeans(n_clusters=2, n_init=1, random_state=0).fit(X)
    mini = MiniBatchKMeans(
        n_clusters=2, batch_size=4, max_iter=5, n_init=1, random_state=0
    ).fit(X)

    assert full.predict(X[:1]).shape == (1,)
    assert mini.predict(X[:1]).shape == (1,)
    _, squared_distances = mini._assign(X, mini.cluster_centers_)
    assert np.isclose(mini.inertia_, np.sum(squared_distances))


def test_kmeans_rejects_zero_iterations():
    with pytest.raises(ValueError, match="max_iter"):
        KMeans(n_clusters=2, max_iter=0).fit(X)


@pytest.mark.parametrize(
    "transformer",
    [SelectKBest(k=1), VarianceThreshold(), Binarizer()],
)
def test_transformers_reject_changed_feature_count(transformer):
    if isinstance(transformer, SelectKBest):
        transformer.fit(X, Y_CLASS)
    else:
        transformer.fit(X)

    with pytest.raises(ValueError, match="different number of features"):
        transformer.transform(np.ones((1, 3)))


def test_perceptron_records_and_checks_feature_count_and_has_score():
    model = Perceptron(max_iter=20, random_state=0).fit(X, Y_CLASS)

    assert model.n_features_in_ == 2
    assert 0.0 <= model.score(X, Y_CLASS) <= 1.0
    with pytest.raises(ValueError, match="different number of features"):
        model.predict(np.ones((1, 3)))


def test_label_encoder_is_cloneable_transformer_and_preserves_axis():
    encoder = LabelEncoder().fit(np.array(["b", "a", "b"]))

    assert isinstance(encoder, BaseTransformer)
    assert isinstance(clone(encoder), LabelEncoder)
    assert encoder.transform(["a"]).shape == (1,)


def test_select_from_model_rejects_wrong_importance_length():
    class BadImportance:
        def fit(self, X, y):
            self.coef_ = np.ones(X.shape[1] + 1)
            return self

    with pytest.raises(ValueError, match="importance length"):
        SelectFromModel(BadImportance()).fit(X, Y_CLASS)


def test_feature_statistics_return_finite_pvalues():
    class_scores, class_pvalues = f_classif(X, Y_CLASS)
    reg_scores, reg_pvalues = f_regression(X, Y_REG)

    assert np.all(np.isfinite(class_pvalues))
    assert np.all(np.isfinite(reg_pvalues))
    assert np.all((class_pvalues >= 0.0) & (class_pvalues <= 1.0))
    assert np.all((reg_pvalues >= 0.0) & (reg_pvalues <= 1.0))
    assert class_scores.shape == reg_scores.shape == (2,)


def test_optimizer_factory_exposes_all_public_optimizers():
    assert isinstance(build_optimizer(OptimizerConfig(name="adagrad")), AdaGrad)
    adamw = build_optimizer(OptimizerConfig(name="adamw", weight_decay=0.2))
    assert isinstance(adamw, AdamW)
    assert adamw.weight_decay == 0.2


def test_sequential_optimizer_keys_are_unique_per_layer():
    class StepLayer:
        def step(self, lr, optimizer, key_prefix):
            value = np.array([1.0])
            optimizer.update(value, value, f"{key_prefix}.W")

    model = Sequential([StepLayer(), StepLayer()], optimizer="adam")
    model.optimizer_ = Adam()
    model._apply_gradients()

    assert set(model.optimizer_._m) == {"layer0.steplayer.W", "layer1.steplayer.W"}


def test_scheduler_internal_typeerror_is_not_swallowed():
    class BrokenScheduler:
        def step(self, metric):
            raise TypeError("scheduler bug")

    model = Sequential([], lr_scheduler=BrokenScheduler())
    with pytest.raises(TypeError, match="scheduler bug"):
        model._on_epoch_end(1, {"monitor_loss": 1.0})


def test_calibration_uses_clones_and_keeps_input_estimator_unfitted():
    base = LogisticRegression(lr=0.1, n_iter=50)
    calibrated = CalibratedClassifierCV(base, cv=2).fit(X, Y_CLASS)

    assert base.coef_ is None
    assert calibrated._calibrated is not base
    assert calibrated.predict_proba(X[:1]).shape == (1, 2)


def test_multioutput_classifier_has_subset_accuracy_score():
    y = np.column_stack([Y_CLASS, 1 - Y_CLASS])
    model = MultiOutputClassifier(LogisticRegression(lr=0.1, n_iter=100)).fit(X, y)

    expected = np.mean(np.all(model.predict(X) == y, axis=1))
    assert np.isclose(model.score(X, y), expected)


def test_modified_huber_is_not_hinge_fallback():
    hinge = SGDClassifier(loss="hinge", max_iter=5, tol=0.0, random_state=0).fit(X, Y_CLASS)
    modified = SGDClassifier(
        loss="modified_huber", max_iter=5, tol=0.0, random_state=0
    ).fit(X, Y_CLASS)

    assert not np.allclose(hinge.coef_, modified.coef_)
