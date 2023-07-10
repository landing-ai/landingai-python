# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project (v1.0.0 and above) adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> WARNING: currently the `landingai` library is in alpha version, and it's not strictly following [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Breaking changes will result a minor version bump instead of a major version bump. Feature releases will result in a patch version bump.

## [0.1.0] - 2023-07-06

### Major changes

- [Support the latest v2 API key](https://github.com/landing-ai/landingai-python/pull/55)
- [Remove support for v1 API key and secret](https://github.com/landing-ai/landingai-python/pull/56)

### Migration Guide

Below section shows you how to fix the backward incompatible changes when you upgrade the version to `0.1.0`.

1. Generate your v2 API key from LandingLens. See [here](https://support.landing.ai/docs/api-key) for more information.
2. The `api_secret` parameter is removed in the `Predictor` and `OcrPredictor` class. `api_key` is a named parameter now, which means you must specify the parameter name, i.e. `api_key`, if you want to pass it to a `Predictor` as an argument.
    See below code as an example:

    **Before upgrade to `0.1.0`**
    ```
    predictor = Predictor(endpoint_id, api_key, api_secret)
    ```
    **After upgrade to `0.1.0`**
    ```
    predictor = Predictor(endpoint_id, api_key=api_key)
    ```
