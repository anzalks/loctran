# Contributing to Loctran

Thank you for your interest in contributing! Here's everything you need to get started.

## Contributor License Agreement (CLA)

By submitting a pull request you agree to the terms of the project's
[Contributor License Agreement](CLA.md). In short, you grant the maintainer
the right to relicense your contribution (including under a commercial
license) while you retain copyright to your work. This is required because
Loctran is dual-licensed (AGPL-3.0 + commercial). No separate signature is
needed — your PR submission is your acceptance.

## Development setup

```bash
git clone https://github.com/anzalks/loctran.git
cd loctran
pip install -e ".[test]"
pip install ruff mypy pre-commit
pre-commit install
```

## Running tests

```bash
pytest -q
```

For coverage:

```bash
pytest --cov=loctran --cov-fail-under=75 -q
```

## Code style

This project uses **ruff** for linting/formatting and **mypy** for type checks. All checks run in CI and must pass before a PR is merged.

```bash
ruff check loctran/
ruff format --check loctran/
mypy loctran/ --ignore-missing-imports
```

## Submitting a PR

**Branch naming**

- `fix/<short-description>` for bug fixes
- `feat/<short-description>` for new features
- `docs/<short-description>` for documentation changes
- `test/<short-description>` for test additions

**PR description template**

```
## What & Why
<!-- One paragraph describing the change and the motivation. -->

## How it works
<!-- Brief technical explanation. -->

## Testing
<!-- What tests were added or changed? -->

## Checklist
- [ ] Tests pass (`pytest -q`)
- [ ] Linting passes (`ruff check loctran/` and `ruff format --check loctran/`)
- [ ] CHANGELOG.md updated
- [ ] Docs updated if behaviour changed
```

## Adding a language or model to the test matrix

The CI matrix is in `.github/workflows/ci.yml`. To extend OS/Python coverage, update the `matrix.os` and `matrix.python-version` sections.

## Updating README screenshots

```bash
pip install -e ".[dev]"
python -m playwright install chromium
make screenshots
```

This writes screenshots to `docs/screenshots/` using `scripts/capture_screenshots.py`.

## Questions?

Open a [GitHub Discussion](https://github.com/anzalks/loctran/discussions) — we read everything.
