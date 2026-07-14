import numpy as np

from mastermlx.neural_net import EarlyStop, History, MLPClassifier, Sequential, Dense, Dropout, OptimizerConfig, TrainingConfig, build_optimizer
from mastermlx.utils import batch_iterator, one_hot, set_seed, shuffle_arrays, shuffle_indices


def test_array_utils_cover_basic_training_needs():
    y = np.array([2, 0, 1, 2])
    oh = one_hot(y, n_classes=3)

    assert oh.shape == (4, 3)
    assert np.array_equal(oh[0], np.array([0.0, 0.0, 1.0]))

    x = np.arange(4)
    z = np.arange(4) * 10
    xs, zs = shuffle_arrays(x, z, random_state=0)
    assert set(xs.tolist()) == set(x.tolist())
    assert np.array_equal(zs // 10, xs)

    batches = list(batch_iterator(x, z, batch_size=3, shuffle=False))
    assert len(batches) == 2
    assert np.array_equal(batches[0][0], np.array([0, 1, 2]))
    assert np.array_equal(shuffle_indices(5, random_state=0), np.array([2, 4, 3, 0, 1]))


def test_set_seed_controls_numpy_global_rng():
    set_seed(42)
    first = np.random.rand(3)
    set_seed(42)
    second = np.random.rand(3)

    assert np.allclose(first, second)


def test_config_objects_and_optimizer_builder():
    cfg = TrainingConfig(n_iter=12, batch_size=4, lr=0.02, validation_split=0.25, patience=3)
    opt_cfg = OptimizerConfig(name="adam", lr=0.005)

    optimizer = build_optimizer(opt_cfg)

    assert cfg.n_iter == 12
    assert cfg.validation_split == 0.25
    assert optimizer.__class__.__name__ == "Adam"


def test_mlp_summary_and_parameter_count():
    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ])
    y = np.array([0, 1, 1, 0])

    model = MLPClassifier(
        hidden_layer_sizes=(3,),
        activation="tanh",
        training_config=TrainingConfig(n_iter=20, lr=0.2, random_state=0, validation_split=0.25, patience=5),
        optimizer_config=OptimizerConfig(name="sgd", lr=0.2),
        random_state=0,
    )
    model.fit(X, y)
    summary = model.summary()

    assert summary["model"] == "MLPClassifier"
    assert summary["num_parameters"] > 0
    assert summary["training_config"].n_iter == 20
    assert len(model.history_) >= 1
    assert model.best_epoch_ is not None


def test_sequential_restores_training_mode_after_validation_and_predict():
    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ])
    y = np.array([0, 1, 1, 0])

    model = Sequential(
        layers=[
            Dense(2, 4, random_state=0),
            Dropout(0.25, random_state=0),
            Dense(4, 2, random_state=1),
        ],
        task="classification",
        training_config=TrainingConfig(n_iter=30, lr=0.2, random_state=0, validation_split=0.25, patience=5),
        optimizer_config=OptimizerConfig(name="sgd", lr=0.2),
        random_state=0,
    )
    model.fit(X, y)
    probs = model.predict_proba(X[:1])

    assert probs.shape == (2,)
    assert model.layers[1].training is True
    assert model.summary()["num_parameters"] == 22
    assert len(model.history_) >= 1
    assert model.best_epoch_ is not None


def test_training_history_and_callback_hooks():
    history = History()
    stop = EarlyStop(monitor="train_loss", patience=2)
    stop.on_train_begin()
    history.on_train_begin()
    for epoch, loss in enumerate([1.0, 0.5, 0.5, 0.5], start=1):
        logs = {"train_loss": loss}
        history.on_epoch_end(epoch, logs)
        stop.on_epoch_end(epoch, logs)

    assert len(history) == 4
    assert history.get("train_loss") == [1.0, 0.5, 0.5, 0.5]
    assert stop.stop_training


def test_nn_clip_norm_is_accepted():
    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ])
    y = np.array([0, 1, 1, 0])

    model = MLPClassifier(
        hidden_layer_sizes=(4,),
        activation="tanh",
        training_config=TrainingConfig(n_iter=15, lr=0.5, clip_norm=0.5, random_state=0, validation_split=0.25, patience=4),
        optimizer_config=OptimizerConfig(name="sgd", lr=0.5),
        random_state=0,
    )
    model.fit(X, y)

    assert model.training_config_.clip_norm == 0.5
    assert model.best_val_loss_ == min(item["val_loss"] for item in model.history_ if item["val_loss"] is not None)
