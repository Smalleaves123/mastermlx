"""Explicit data contracts for tabular training and inference boundaries."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from .quality import _as_table, _is_numeric, _key, _missing_mask, quality_report
from .schema import compare_schema


class DataContract:
    """Validate tabular data against a fitted schema and explicit column rules.

    ``rules`` maps column names to dictionaries such as
    ``{"kind": "numeric", "min": 0, "max": 120, "max_missing_rate": 0.1}``
    or ``{"kind": "categorical", "allowed_values": ["cn", "us"]}``.
    The contract records the training schema in :meth:`fit` and returns a
    structured validation report instead of silently coercing bad input.
    """

    def __init__(
        self,
        rules=None,
        *,
        required_columns=None,
        max_missing_rate=None,
        unique_columns=None,
        allow_extra_columns=False,
        strict_dtypes=False,
    ):
        self.rules = {} if rules is None else {str(name): dict(rule) for name, rule in dict(rules).items()}
        self.required_columns = None if required_columns is None else [str(name) for name in required_columns]
        self.max_missing_rate = max_missing_rate
        self.unique_columns = set() if unique_columns is None else {str(name) for name in unique_columns}
        self.allow_extra_columns = bool(allow_extra_columns)
        self.strict_dtypes = bool(strict_dtypes)
        self.reference_ = None
        self.reference_quality_ = None
        self.feature_names_ = None

        if max_missing_rate is not None and not 0.0 <= float(max_missing_rate) <= 1.0:
            raise ValueError("max_missing_rate must be between 0 and 1")
        for name, rule in self.rules.items():
            if not isinstance(rule, dict):
                raise TypeError(f"rule for {name!r} must be a dictionary")
            rate = rule.get("max_missing_rate")
            if rate is not None and not 0.0 <= float(rate) <= 1.0:
                raise ValueError(f"max_missing_rate for {name!r} must be between 0 and 1")
            kind = rule.get("kind")
            if kind is not None and kind not in {"numeric", "categorical"}:
                raise ValueError(f"kind for {name!r} must be 'numeric' or 'categorical'")

    def fit(self, X, y=None):
        """Record the reference schema and validate the contract definition."""

        _, names = _as_table(X)
        required = names if self.required_columns is None else self.required_columns
        missing_required = [name for name in required if name not in names]
        missing_rules = [name for name in self.rules if name not in names]
        if missing_required:
            raise ValueError(f"required columns are missing from fit data: {missing_required}")
        if missing_rules:
            raise ValueError(f"rule columns are missing from fit data: {missing_rules}")
        unknown_unique = sorted(self.unique_columns.difference(names))
        if unknown_unique:
            raise ValueError(f"unique columns are missing from fit data: {unknown_unique}")

        self.reference_ = deepcopy(X)
        self.feature_names_ = list(names)
        self.required_columns = list(required)
        self.reference_quality_ = quality_report(X, y)
        return self

    def _require_fitted(self):
        if self.reference_ is None or self.feature_names_ is None:
            raise RuntimeError("DataContract has not been fit yet")

    @staticmethod
    def _add_issue(issues, code, column, message, **details):
        item = {"code": code, "column": column, "message": message}
        item.update(details)
        issues.append(item)

    def validate(self, X):
        """Return a structured report with ``valid``, ``errors``, and ``warnings``."""

        self._require_fitted()
        test_arr, test_names = _as_table(X)
        test_map = {name: index for index, name in enumerate(test_names)}
        schema = compare_schema(self.reference_, X)
        errors: list[dict[str, object]] = []
        warnings: list[dict[str, object]] = []
        required_columns = list(self.required_columns or [])
        feature_names = list(self.feature_names_ or [])

        missing = [name for name in required_columns if name not in test_map]
        if missing:
            self._add_issue(errors, "missing_column", None, "required columns are missing", columns=missing)
        extra = [name for name in test_names if name not in feature_names]
        if extra and not self.allow_extra_columns:
            self._add_issue(errors, "extra_column", None, "unexpected columns are present", columns=extra)
        if not schema["order_match"]:
            self._add_issue(warnings, "column_order", None, "column order differs from the reference schema")
        for change in schema["dtype_changes"]:
            issue_list = errors if self.strict_dtypes else warnings
            self._add_issue(
                issue_list,
                "dtype_change",
                change["name"],
                "column dtype differs from the reference schema",
                reference=change["train"],
                observed=change["test"],
            )

        for name, rule in self.rules.items():
            if name not in test_map:
                continue
            column = test_arr[:, test_map[name]]
            missing_mask = _missing_mask(column)
            values = column[~missing_mask]
            missing_rate = float(missing_mask.mean())
            max_rate = rule.get("max_missing_rate", self.max_missing_rate)
            if max_rate is not None and missing_rate > float(max_rate):
                self._add_issue(
                    errors,
                    "missing_rate",
                    name,
                    "missing rate exceeds the contract limit",
                    observed=missing_rate,
                    limit=float(max_rate),
                )

            kind = "numeric" if _is_numeric(column, missing_mask) else "categorical"
            expected_kind = rule.get("kind")
            if expected_kind is not None and kind != expected_kind:
                self._add_issue(
                    errors,
                    "kind_change",
                    name,
                    "column kind differs from the contract",
                    expected=expected_kind,
                    observed=kind,
                )

            if kind == "numeric" and values.size:
                numeric = np.asarray(values, dtype=float)
                finite = numeric[np.isfinite(numeric)]
                if rule.get("min") is not None and finite.size and np.min(finite) < float(rule["min"]):
                    self._add_issue(
                        errors,
                        "range_min",
                        name,
                        "value is below the contract minimum",
                        observed=float(np.min(finite)),
                        limit=float(rule["min"]),
                    )
                if rule.get("max") is not None and finite.size and np.max(finite) > float(rule["max"]):
                    self._add_issue(
                        errors,
                        "range_max",
                        name,
                        "value exceeds the contract maximum",
                        observed=float(np.max(finite)),
                        limit=float(rule["max"]),
                    )

            if rule.get("allowed_values") is not None:
                allowed = {_key(value) for value in rule["allowed_values"]}
                unseen = [value for value in values if _key(value) not in allowed]
                if unseen:
                    self._add_issue(
                        errors,
                        "allowed_values",
                        name,
                        "column contains values outside the allowed set",
                        unseen_values=[value.item() if isinstance(value, np.generic) else value for value in unseen[:5]],
                    )

            if rule.get("unique", False) or name in self.unique_columns:
                keys = {_key(value) for value in values}
                if len(keys) != values.size:
                    self._add_issue(errors, "unique", name, "column contains duplicate non-missing values")

        return {
            "valid": not errors,
            "errors": errors,
            "warnings": warnings,
            "schema": schema,
            "checked_columns": list(test_names),
        }

    def check(self, X, *, raise_on_error=False):
        """Validate data and optionally raise on the first contract failure."""

        report = self.validate(X)
        if raise_on_error and not report["valid"]:
            first = report["errors"][0]
            raise ValueError(f"data contract violation: {first['message']}")
        return report

    def summary(self):
        """Return JSON-friendly contract metadata."""

        self._require_fitted()
        required_columns = list(self.required_columns or [])
        feature_names = list(self.feature_names_ or [])
        return {
            "required_columns": required_columns,
            "feature_names": feature_names,
            "rules": deepcopy(self.rules),
            "unique_columns": sorted(self.unique_columns),
            "allow_extra_columns": self.allow_extra_columns,
            "strict_dtypes": self.strict_dtypes,
        }


__all__ = ["DataContract"]
