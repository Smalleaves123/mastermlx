"""Command-line inspection and inference for mastermlx model bundles."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

from .base import ModelBundle
from .base.results import to_json_safe


class _InputTable:
    def __init__(self, rows, columns):
        self.values = np.asarray(rows, dtype=object)
        self.columns = list(columns)

    def __array__(self, dtype=None, copy=None):
        result = np.asarray(self.values, dtype=dtype)
        return result.copy() if copy else result


def _feature_names(bundle):
    names = bundle.feature_names_in_
    return None if names is None else [str(name) for name in names]


def _column_kinds(bundle):
    kinds = {}
    contract = bundle.data_contract_
    if contract is not None:
        for name, rule in contract.rules.items():
            if rule.get("kind") in {"numeric", "categorical"}:
                kinds[str(name)] = rule["kind"]
    preprocessor = bundle.preprocessing_
    for attr, kind in (("numeric_cols_", "numeric"), ("categorical_cols_", "categorical")):
        values = getattr(preprocessor, attr, None)
        if values is None:
            continue
        names = _feature_names(bundle) or []
        for value in np.asarray(values, dtype=object).ravel():
            if isinstance(value, (int, np.integer)) and 0 <= int(value) < len(names):
                kinds[names[int(value)]] = kind
            else:
                kinds[str(value)] = kind
    return kinds


def _parse_csv_value(value, kind):
    if value is None:
        return None
    stripped = value.strip()
    if stripped.lower() in {"", "null", "none", "nan"}:
        return None
    if kind == "categorical":
        return stripped
    if kind == "numeric":
        try:
            return float(stripped)
        except ValueError as exc:
            raise ValueError(f"expected a numeric value, got {value!r}") from exc
    try:
        return float(stripped)
    except ValueError:
        return stripped


def _load_csv(path, bundle):
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames
        if not columns:
            raise ValueError("CSV input must contain a header row")
        raw_rows = list(reader)
    if not raw_rows:
        raise ValueError("input must contain at least one row")
    kinds = _column_kinds(bundle)
    rows = [
        [_parse_csv_value(row[name], kinds.get(name)) for name in columns]
        for row in raw_rows
    ]
    return _InputTable(rows, columns)


def _load_json(path, bundle):
    payload = json.loads(path.read_text())
    columns = None
    if isinstance(payload, dict):
        columns = payload.get("columns")
        payload = payload.get("records", payload.get("rows"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("JSON input must contain a non-empty records or rows list")
    if isinstance(payload[0], dict):
        columns = _feature_names(bundle) if columns is None else list(columns)
        if columns is None:
            columns = list(payload[0])
        missing = [name for name in columns if any(name not in row for row in payload)]
        if missing:
            raise ValueError(f"JSON records are missing columns: {sorted(set(missing))}")
        rows = [[row[name] for name in columns] for row in payload]
    else:
        rows = payload
        columns = _feature_names(bundle) if columns is None else list(columns)
        if columns is None:
            columns = [f"x{index}" for index in range(len(rows[0]))]
    if any(not isinstance(row, list) or len(row) != len(columns) for row in rows):
        raise ValueError("JSON rows must have one value per column")
    return _InputTable(rows, columns)


def _load_input(path, bundle, input_format):
    resolved = input_format
    if resolved == "auto":
        resolved = path.suffix.lower().lstrip(".")
    if resolved == "csv":
        return _load_csv(path, bundle)
    if resolved == "json":
        return _load_json(path, bundle)
    raise ValueError("input format must be json or csv")


def _prediction_payload(bundle, table, include_probabilities):
    prediction = np.asarray(bundle.predict(table))
    result = {
        "format": "mastermlx-predictions",
        "n_samples": int(prediction.shape[0]),
        "predictions": to_json_safe(prediction),
    }
    if include_probabilities:
        result["probabilities"] = to_json_safe(np.asarray(bundle.predict_proba(table)))
    return result


def _prediction_rows(payload):
    predictions = np.asarray(payload["predictions"], dtype=object)
    if predictions.ndim == 1:
        rows = [{"prediction": predictions[index]} for index in range(predictions.shape[0])]
    else:
        rows = [
            {f"prediction_{column}": value for column, value in enumerate(predictions[index])}
            for index in range(predictions.shape[0])
        ]
    probabilities = payload.get("probabilities")
    if probabilities is not None:
        probabilities = np.asarray(probabilities, dtype=float)
        if probabilities.ndim == 1:
            probabilities = probabilities[:, None]
        for index, row in enumerate(rows):
            for column, value in enumerate(probabilities[index]):
                row[f"probability_{column}"] = float(value)
    return rows


def _write_payload(payload, output, output_format):
    if output is None:
        sys.stdout.write(json.dumps(to_json_safe(payload), indent=2, sort_keys=True) + "\n")
        return
    resolved = output_format
    if resolved == "auto":
        resolved = output.suffix.lower().lstrip(".") or "json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if resolved == "json":
        output.write_text(json.dumps(to_json_safe(payload), indent=2, sort_keys=True) + "\n")
        return
    if resolved != "csv":
        raise ValueError("output format must be json or csv")
    if "predictions" in payload:
        rows = _prediction_rows(payload)
    else:
        rows = []
        for name, value in to_json_safe(payload).items():
            if isinstance(value, (dict, list)):
                value = json.dumps(value, sort_keys=True)
            rows.append({"name": name, "value": value})
    if not rows:
        raise ValueError("prediction output is empty")
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="print a bundle model card")
    inspect_parser.add_argument("bundle", type=Path)
    inspect_parser.add_argument("--output", type=Path)

    for command in ("validate", "predict"):
        command_parser = subparsers.add_parser(command, help=f"{command} an input batch")
        command_parser.add_argument("bundle", type=Path)
        command_parser.add_argument("input", type=Path)
        command_parser.add_argument(
            "--input-format", choices=("auto", "json", "csv"), default="auto"
        )
        command_parser.add_argument("--output", type=Path)
        command_parser.add_argument(
            "--output-format", choices=("auto", "json", "csv"), default="auto"
        )
        if command == "predict":
            command_parser.add_argument("--probabilities", action="store_true")
    return parser


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        bundle = ModelBundle.load(args.bundle)
        if args.command == "inspect":
            _write_payload(bundle.model_card(), args.output, "json")
            return 0
        table = _load_input(args.input, bundle, args.input_format)
        if args.command == "validate":
            report = bundle.validate(table)
            _write_payload(report, args.output, args.output_format)
            return 0 if report["valid"] else 1
        payload = _prediction_payload(bundle, table, args.probabilities)
        _write_payload(payload, args.output, args.output_format)
        return 0
    except (AttributeError, OSError, TypeError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
