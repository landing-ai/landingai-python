# Developer Guide

This guide introduces you to the `landingai` development process and provides information on writing, testing, and building the `landingai` library.

Read this guide if you need to perform any of the following tasks:

- Install the `landingai` library locally.
- Contribute to the `landingai` library.

## Install `landingai` Library Locally

### Prerequisite: Install Poetry

> `landingai` uses `Poetry` for packaging and dependency management. If you want to build it from source, you have to install Poetry first. To see all possible options, refer to the [Poetry documentation](https://python-poetry.org/docs/#installation).

For Linux, macOS, Windows (WSL):

```
curl -sSL https://install.python-poetry.org | python3 -
```

Note: You can switch to use a different Python version by specifying the python version:

```
curl -sSL https://install.python-poetry.org | python3.10 -
```

Or run the following command after installing Poetry:

```
poetry env use 3.10
```

### Install All Dependencies

```bash
poetry install --all-extras
```

### Activate the virtualenv

```bash
poetry shell
```

## Test and Lint `landingai`

### Run Linting

```bash
poetry run flake8 . --exclude .venv --count --show-source --statistics
```

### Run Tests

```bash
poetry run pytest tests/
```

## Release

The CI and CD pipelines are defined in the `.github/workflows/ci_cd.yml` file.

Every git commit will trigger a release to `PyPi` at https://pypi.org/project/landingai/

### Versioning

When we release a new library version, we version it using [Sematic Versioning 2.0.0](https://semver.org/)(`MAJOR.MINOR.PATCH`). The version number is defined in the `pyproject.toml` file in the `version` field.

As a general rule of thumb, given a version number `MAJOR.MINOR.PATCH`, increment the:

- `MAJOR` version when you make incompatible API changes.
- `MINOR` version when you add functionality in a backward-compatible manner, such as adding a new feature.
- `PATCH` version when you make backward-compatible bug fixes and minor changes.

Note: The CD pipeline will automatically increment the `PATCH` version for every git commit.
**For a `MINOR` or `MAJOR` version change, you need to manually update `pyproject.toml` to bump the version number.**
