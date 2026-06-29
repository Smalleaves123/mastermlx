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
pytest tests/
```

## 4. Verify install from artifacts

Prefer testing both the source distribution and the wheel in a clean environment.

## 5. Publish

- Upload the distribution artifacts to PyPI
- Create a Git tag that matches the release version
- Create a GitHub release with a short summary and links

### GitHub Actions path

- Push a tag like `v0.1.1` to trigger `.github/workflows/release.yml`
- For a dry run, open the `Release` workflow in GitHub Actions and run `workflow_dispatch` with `target=testpypi`
- Store the following repository secrets:
  - `PYPI_API_TOKEN`
  - `TEST_PYPI_API_TOKEN`
- The workflow builds distributions, checks them with `twine`, publishes to the selected index, and creates a GitHub Release for tag pushes

## 6. Post-release

- Confirm installation with `pip install mastermlx`
- Smoke test the top-level import and one or two core models
