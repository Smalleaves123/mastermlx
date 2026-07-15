import numpy as np
import pytest
import zipfile

from mastermlx.neural_net import (
    Dense,
    ModelCheckpoint,
    Sequential,
    TrainingConfig,
)


def test_sequential_multilabel_metrics_and_accumulation(tmp_path):
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    model = Sequential(
        [Dense(2, 6, random_state=0), Dense(6, 2, random_state=1)],
        task="multilabel",
        optimizer="adam",
        lr=0.1,
        training_config=TrainingConfig(
            n_iter=30,
            lr=0.1,
            accumulation_steps=2,
            metrics=("accuracy", "f1"),
            random_state=0,
        ),
    )
    model.fit(X, y)

    assert model.predict(X).shape == y.shape
    assert "train_accuracy" in model.history_[-1]
    assert "train_f1" in model.history_[-1]


def test_model_checkpoint_roundtrips_full_training_object(tmp_path):
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 1, 1, 0])
    path = tmp_path / "model.checkpoint"
    callback = ModelCheckpoint(path, monitor="train_loss", save_best_only=False)
    model = Sequential(
        [Dense(2, 4, random_state=0), Dense(4, 2, random_state=1)],
        optimizer="adam",
        lr=0.05,
        n_iter=5,
        random_state=0,
        callbacks=[callback],
    )
    model.fit(X, y)
    restored = Sequential.load_checkpoint(path)

    assert path.exists()
    assert np.array_equal(restored.predict(X), model.predict(X))
    assert restored.optimizer_._t == model.optimizer_._t
    previous_step = restored.optimizer_._t
    restored.fit(X, y, resume=True)
    assert restored.optimizer_._t > previous_step


def test_checkpoint_has_versioned_safe_manifest(tmp_path):
    model = Sequential([Dense(2, 2, random_state=0)], n_iter=1, task="classification")
    model.fit(np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([0, 1]))
    path = tmp_path / "safe.checkpoint"
    model.save_checkpoint(path)

    with zipfile.ZipFile(path) as archive:
        manifest = archive.read("manifest.json").decode("utf-8")
    assert '"format":"mastermlx-checkpoint"' in manifest
    assert '"schema_version":1' in manifest
    assert '"state_schema":{"name":"object_graph","version":1}' in manifest


def test_legacy_checkpoint_format_is_rejected(tmp_path):
    path = tmp_path / "legacy.checkpoint"
    path.write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match="unsupported checkpoint format"):
        Sequential.load_checkpoint(path)
