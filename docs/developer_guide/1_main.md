# Developer Guide

This guide introduces you to the `landingai` development process and provides information on writing, testing, and building the `landingai` library.

Read this guide if you need to perform one of the below tasks:

1. Install the `landingai` library locally.
2. Contribute to the `landingai` library.

## Install `landingai` Library Locally

### Prerequisite - Install Poetry

> `landingai` uses `Poetry` for packaging and dependency management. If you want to build it from source, you have to install Poetry first. Please follow [the official guide](https://python-poetry.org/docs/#installation) to see all possible options.

For Linux, macOS, Windows (WSL):

```
curl -sSL https://install.python-poetry.org | python3 -
```

NOTE: you can switch to use a different Python version by specifying the python version:

```
curl -sSL https://install.python-poetry.org | python3.10 -
```

or run below command after you have installed poetry:

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

The CI & CD pipeline is defined in the `.github/workflows/ci_cd.yml` file.

Every git commit will trigger a release to `PyPi` at https://pypi.org/project/landingai/

### Versioning

We follow [Sematic Versioning 2.0.0](https://semver.org/), i.e. `MAJOR.MINOR.PATCH`, when release a new library version. The version number is defined in the `pyproject.toml` file by the `version` field.

General rule of thumb, given a version number `MAJOR.MINOR.PATCH`, increment the:

1. `MAJOR` version when you make incompatible API changes.
2. `MINOR` version when you add functionality in a backward compatible manner, e.g. adding a new feature.
3. `PATCH` version when you make backward compatible bug fixes and minor changes.

NOTE: the CD pipeline will automatically increment the `PATCH` version for every git commit.
**For a `MINOR` or `MAJOR` version change, you need to manually update `pyproject.toml` to bump the version number.**
