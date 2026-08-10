from __future__ import annotations

import numpy as np


def feature_importances(estimator, n_features):
    """Return one non-negative importance per input feature."""

    if hasattr(estimator, "coef_"):
        coefficients = np.asarray(estimator.coef_, dtype=float)
        if coefficients.ndim == 1:
            importance = np.abs(coefficients)
        elif coefficients.ndim == 2:
            matching_axes = [
                axis for axis, size in enumerate(coefficients.shape) if size == n_features
            ]
            if not matching_axes:
                raise ValueError(
                    "estimator coefficient dimensions do not contain the input feature count"
                )
            if len(matching_axes) == 1:
                feature_axis = matching_axes[0]
            else:
                # mastermlx multiclass linear estimators store
                # (n_features, n_classes); sklearn-style estimators generally
                # store (n_classes, n_features).
                module = type(estimator).__module__
                feature_axis = 0 if module.startswith("mastermlx.") else 1
            reduce_axis = 1 - feature_axis
            importance = np.sum(np.abs(coefficients), axis=reduce_axis)
        else:
            raise ValueError("estimator coef_ must be one- or two-dimensional")
    elif hasattr(estimator, "feature_importances_"):
        importance = np.asarray(estimator.feature_importances_, dtype=float).ravel()
    else:
        raise ValueError("estimator must have coef_ or feature_importances_")

    importance = np.asarray(importance, dtype=float).ravel()
    if importance.size != n_features:
        raise ValueError(
            "estimator feature importance length must match X "
            f"({importance.size} != {n_features})"
        )
    return importance
