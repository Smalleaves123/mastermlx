# API stability

`mastermlx` classifies public APIs into four lifecycle tiers:

- **Stable**: compatibility is preserved across minor releases.
- **Beta**: suitable for real use, but signatures may still evolve with a
  documented migration path.
- **Experimental**: evaluation-stage APIs that may change between minors.
- **Internal**: private implementation details with no compatibility promise.

Query a name without importing its implementation package:

```python
from mastermlx.api import get_api_stability

assert get_api_stability("KMeans") == "stable"
assert get_api_stability("robotics.RobotModel") == "experimental"
```

`api_stability_report()` returns every top-level public name grouped by tier.
Package defaults live in `mastermlx.api.PACKAGE_STABILITY`; exceptional names
are listed in `STABILITY_OVERRIDES`. A new public export must be covered by one
of those rules and by `tests/test_api_stability_bundle.py`.

Promoting an API requires contract tests, public documentation, and an entry
in `CHANGELOG.md`. Demotion requires a deprecation period under the rules in
[`api_policy.md`](api_policy.md).

Representative Stable estimators and transformers are continuously checked by
`tests/test_stable_api_contracts.py`. The matrix enforces cloneability, fitted
feature metadata, single-sample output shapes, incompatible-feature rejection,
incremental-learning signatures, multi-output regression, and sparse input
validation across the stable estimator families.
