# Model bundles

`ModelBundle` combines the validation, preprocessing, and estimator stages of
tabular inference into one safely serializable object.

```python
from mastermlx import DataContract, LogisticRegression, ModelBundle

contract = DataContract(
    rules={"age": {"kind": "numeric", "min": 0, "max": 120}},
)
bundle = ModelBundle(
    LogisticRegression(random_state=0),
    preprocessing="auto",
    data_contract=contract,
    metadata={"owner": "risk-team"},
).fit(X_train, y_train)

bundle.save("customer-risk.mlx")
restored = ModelBundle.load("customer-risk.mlx")
prediction = restored.predict(X_new)
```

An existing tabular experiment can package its already-fitted best pipeline
without another training pass:

```python
experiment.fit(X_train, y_train)
bundle = experiment.to_bundle(metadata={"owner": "risk-team"})
bundle.save("selected-model.mlx")
```

Inference checks the original feature names and order, then applies the fitted
data contract before preprocessing. Use `validate(X)` to inspect a structured
report without running prediction, and `summary()` or `export_summary()` for
JSON-safe deployment metadata.

## Model cards and command-line inference

Put governance details such as `owner`, `task`, `intended_use`, `limitations`,
`metrics`, and `training_data` in bundle metadata. `model_card()` turns them
into a structured report and lists missing governance fields rather than
inventing values.

Installed packages expose a dependency-free command-line interface:

```bash
mastermlx inspect customer-risk.mlx
mastermlx validate customer-risk.mlx incoming.csv
mastermlx predict customer-risk.mlx incoming.json --probabilities
mastermlx predict customer-risk.mlx incoming.csv --output predictions.csv
```

JSON input accepts `records` (objects) or `rows` plus `columns`. CSV files must
contain a header. Output defaults to JSON on standard output and switches to
CSV when the output filename ends in `.csv`.

Artifacts use the existing versioned `mastermlx-checkpoint` format. Arrays are
loaded with `allow_pickle=False`; categorical object arrays are represented as
recursively validated values rather than pickle payloads. Fitted data
contracts retain a schema and quality summary, but not the original training
rows.
