from __future__ import annotations

import numpy as np

from ..utils.array import batch_iterator


def run_supervised_training_loop(
    *,
    X_train,
    y_train,
    X_val,
    y_val,
    batch_size,
    n_iter,
    shuffle,
    random_state,
    tol,
    patience,
    verbose,
    train_step,
    evaluate_train_loss,
    evaluate_val_loss=None,
    snapshot_state=None,
    restore_state=None,
):
    rng = np.random.default_rng(random_state)
    loss_history = []
    val_loss_history = []
    history = []
    best_loss = np.inf
    best_epoch = None
    best_val_loss = None
    best_state = None
    wait = 0

    for epoch in range(int(n_iter)):
        for xb, yb in batch_iterator(
            X_train,
            y_train,
            batch_size=batch_size,
            shuffle=shuffle,
            random_state=rng.integers(0, 1 << 31),
        ):
            train_step(xb, yb)

        train_loss = float(evaluate_train_loss(X_train, y_train))
        loss_history.append(train_loss)

        monitor_loss = train_loss
        val_loss = None
        if X_val is not None and evaluate_val_loss is not None:
            val_loss = float(evaluate_val_loss(X_val, y_val))
            val_loss_history.append(val_loss)
            monitor_loss = val_loss

        improved = monitor_loss + tol < best_loss
        if improved:
            best_loss = monitor_loss
            best_epoch = epoch + 1
            best_val_loss = monitor_loss
            if X_val is not None and snapshot_state is not None:
                best_state = snapshot_state()
            wait = 0
        else:
            wait += 1

        if verbose:
            if val_loss is None:
                print(f"epoch={epoch + 1} loss={train_loss:.6f}")
            else:
                print(f"epoch={epoch + 1} loss={train_loss:.6f} val_loss={monitor_loss:.6f}")

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "val_loss": None if val_loss is None else float(val_loss),
            }
        )

        if patience is not None and wait >= int(patience):
            break
        if X_val is None and epoch > 0 and len(loss_history) >= 2 and abs(loss_history[-2] - loss_history[-1]) < tol:
            break

    if best_state is not None and restore_state is not None:
        restore_state(best_state)

    return {
        "loss": loss_history,
        "val_loss": val_loss_history,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }
