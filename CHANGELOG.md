# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project (v1.0.0 and above) adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> WARNING: currently the `landingai` library is in alpha version, and it's not strictly following [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Breaking changes will result a minor version bump instead of a major version bump. Feature releases will result in a patch version bump.

## [0.3.0] - 2023-XX-YY

### Major changes

- [Image source iterators yielding single Frame](https://github.com/landing-ai/landingai-python/pull/125)


### Migration Guide

Whenever you iterate over an image source (`NetworkedCamera`, `Webcam`, etc), each iteration yields a single `Frame`, and not a `FrameSet`.
But most `FrameSet` operations were migrated to `Frame` class, so you can still use the same API to manipulate the `Frame` object with very minor changes:

1. On `for frame in image_source:`, don't use `frame.frames[0]` anymore. Instead, you should just use the `frame` object directly (to do resize, check predictions, overlay predictions, etc).
2. `Frame.save_image` receives a direct file path to where the frame image should be saved (not just a prefix, as it happens with `FrameSet`).



## [0.2.0] - 2023-07-12

### Major changes

- [Refactor visual pipeline functionality](https://github.com/landing-ai/landingai-python/pull/77)

### Migration Guide

1. The `landingai.vision_pipeline` module was migrated to `landingai.pipeline.FrameSet`
2. All the image sources were consolidated under `landingai.pipeline.image_source` in particular `NetworkedCamera`
3. `read_file` is now in `landingai.storage.data_access` and now returns a dictionary. The file contents can be found under "content".

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
