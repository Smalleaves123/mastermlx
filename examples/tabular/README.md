# Tabular Readiness Tutorial

[`readiness_demo.py`](readiness_demo.py) shows the business-facing readiness
workflow in 0.1.15:

```bash
python examples/tabular/readiness_demo.py
```

It fits a reference profile, validates incoming data against a data contract,
and prints readiness status plus concrete issue labels.

The high-level workflow returns dictionary-compatible report objects. They
support field access, `as_dict(json_safe=True)`, `to_json(path)`, and
`to_csv(path)`:

```python
from mastermlx.tabular import DataReadinessReport

readiness = DataReadinessReport(data_contract=contract)
report = readiness.fit(reference).run(candidate)
print(report.status)
report.to_json("readiness.json")
```

See [`docs/workflows.md`](../../docs/workflows.md) for the shared experiment
and report contract, and the [`0.1.15 API guide`](../API_REFERENCE.md) for
lower-level preprocessing and evaluation tools.
