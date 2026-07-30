"""Shared result and experiment containers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def to_json_safe(value: Any) -> Any:
    """Convert common NumPy and path objects into JSON-serializable values."""

    if isinstance(value, np.ndarray):
        return to_json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


class BaseResult(dict[str, Any]):
    """Dictionary-compatible result object with attribute access and export helpers."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def as_dict(self, *, json_safe: bool = False) -> dict[str, Any]:
        """Return a plain dictionary copy.

        Set ``json_safe=True`` when the result may contain NumPy arrays,
        scalars, tuples, or paths that need conversion before serialization.
        """

        data = dict(self)
        return to_json_safe(data) if json_safe else data

    def to_json(self, path=None, *, indent: int = 2, sort_keys: bool = True) -> str:
        """Serialize the result to JSON and optionally write it to ``path``."""

        text = json.dumps(self.as_dict(json_safe=True), indent=indent, sort_keys=sort_keys)
        if path is not None:
            Path(path).write_text(text + "\n")
        return text

    def to_rows(self) -> list[dict[str, Any]]:
        """Return a row-oriented representation for compact CSV exports."""

        rows = []
        for key, value in self.as_dict(json_safe=True).items():
            if isinstance(value, (dict, list)):
                value = json.dumps(value, sort_keys=True)
            rows.append({"name": key, "value": value})
        return rows

    def to_csv(self, path, *, fieldnames=("name", "value")):
        """Write a compact name/value CSV export and return the path."""

        output = Path(path)
        rows = self.to_rows()
        with output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
            writer.writeheader()
            writer.writerows(rows)
        return output


class BaseReport(BaseResult):
    """Semantic alias for report-shaped result objects."""


class BaseExperiment:
    """Small mixin for workflow classes that produce reports and artifacts."""

    def __init__(self):
        self.report_ = None
        self.artifacts_ = BaseResult()

    def _store_report(self, report):
        self.report_ = report if isinstance(report, BaseResult) else BaseReport(report)
        return self.report_

    def _store_artifact(self, name: str, value):
        self.artifacts_[str(name)] = value
        return value

    def export_report(self, path, report=None):
        """Export a report to JSON and return the output path."""

        result = self.report_ if report is None else report
        if result is None:
            raise RuntimeError("experiment has no report to export")
        result = result if isinstance(result, BaseResult) else BaseReport(result)
        result.to_json(path)
        return self._store_artifact("report_json", Path(path))


def export_reports(reports, directory, *, manifest_name="manifest.json"):
    """Export a mapping of report names to JSON files plus a manifest."""

    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = BaseResult()
    for name, report in dict(reports).items():
        safe_name = str(name).strip().replace(" ", "_").replace("/", "_")
        if not safe_name:
            raise ValueError("report names must be non-empty")
        result = report if isinstance(report, BaseResult) else BaseReport(report)
        path = output_dir / f"{safe_name}.json"
        result.to_json(path)
        artifacts[safe_name] = path
    manifest = BaseReport(
        {
            "directory": output_dir,
            "reports": {name: str(path) for name, path in artifacts.items()},
            "n_reports": len(artifacts),
        }
    )
    manifest_path = output_dir / manifest_name
    manifest.to_json(manifest_path)
    artifacts["manifest"] = manifest_path
    return artifacts


__all__ = ["BaseExperiment", "BaseReport", "BaseResult", "export_reports", "to_json_safe"]
