# Releasing mastermlx

Use this checklist before publishing a new version.

## 1. Update version

- Bump `mastermlx/version.py`
- Bump `pyproject.toml` project version
- Update `CHANGELOG.md`

## 2. Verify packaging metadata

```bash
python -m pip install --upgrade build twine
python -m build
twine check dist/*
```

## 3. Run tests

```bash
python -m pytest tests/
MPLBACKEND=Agg python scripts/run_examples_smoke.py
python scripts/check_import_budget.py --budget-ms 250 --runs 5
python scripts/generate_lazy_exports.py --check
```

## 4. Verify install from artifacts

Prefer testing both the source distribution and the wheel in a clean environment.

## 5. Publish

The single [`Release`](.github/workflows/release.yml) workflow owns build,
validation, publication, and GitHub release creation. Do not upload artifacts
or create a release separately.

### TestPyPI dry run

Run the `Release` workflow manually with `target=testpypi`. It builds Linux,
macOS, and Windows wheels plus the source distribution, runs wheel and sdist
smoke tests, then uploads the exact validated artifacts to TestPyPI.

### PyPI release

1. Commit the version and changelog updates.
2. Push a matching tag such as `v0.1.16`.
3. The workflow builds and validates all artifacts, publishes them to PyPI, and
   creates the GitHub Release only after publication succeeds.

Manual `target=pypi` publication is available for recovery, but normal releases
should use an annotated version tag.

Store these repository secrets until trusted publishing is configured:

- `PYPI_TOKEN`
- `TEST_PYPI_API_TOKEN`

## 6. Post-release

- Confirm installation with `pip install mastermlx==<version>` in a clean
  environment.
- Smoke test the lazy top-level import, one core model, and one compiled path.
- Confirm the GitHub Release contains all validated wheels and the source
  distribution.
