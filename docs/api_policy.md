# API Policy

`mastermlx` keeps a broad top-level namespace for convenience, but new public
surfaces should be categorized before release.

## Stability Levels

- Stable: documented names used in README, docs, or examples; preserve behavior
  across minor releases where practical.
- Experimental: useful workflow or algorithm APIs that may still change; note
  the status in docs or docstrings.
- Deprecated: supported temporarily with a changelog entry and replacement path.
- Internal: private helpers, compiled kernels, and implementation modules.

## Public Export Rules

- Add names to a package `__all__` only when they are intended for public use.
- Prefer domain package exports first, then top-level exports for stable names.
- Keep compatibility aliases when a name has already appeared in examples,
  release notes, or tests.
- Public result objects should be dictionary-compatible.

## Deprecation Rules

When replacing a public API, keep the old import path for at least one minor
release when possible. Add tests for the compatibility path and document the
replacement in `CHANGELOG.md`.
