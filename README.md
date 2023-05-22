<p align="center">
  <img width="200" height="200" src="https://github.com/landing-ai/landingai-python/raw/main/assets/avi-logo.png">
</p>

# LandingLens code sample repository
This repository contains LandingLens development library and running examples showing how to integrate LandingLens on a variety of scenarios. All the examples show different ways to acquire images from multiple sources and techniques to process the results. Jupyter notebooks focus on ease of use while Python apps include more robust and complete examples.

<!-- Generated using https://www.tablesgenerator.com/markdown_tables -->

| example | description | language |
|---|---|---|
| [Poker card suit identification](https://github.com/landing-ai/landingai-python/blob/main/examples/webcam-collab-notebook/webcam-collab-notebook.ipynb) | This notebook can run directly in Google collab using the web browser camera to detect suits on poker card | Jupyter Notebook [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/landing-ai/landingai-python/blob/main/examples/webcam-collab-notebook/webcam-collab-notebook.ipynb)|
| [Door monitoring for home automation](https://github.com/landing-ai/landingai-python/blob/main/examples/rtsp-capture-notebook/rtsp-capture.ipynb) | This notebook uses an object detection model to determine whether a door is open or closed. The notebook can acquire images directly from an RTSP camera | Jupyter Notebook |
| [Streaming capture service](https://github.com/landing-ai/landingai-python/tree/main/examples/capture-service) | This application shows how to do continuous acquisition from an image sensor using RTSP. | Python application |
| [Pixel coverage post-processing](https://github.com/landing-ai/landingai-python/tree/main/examples/post-processings/farmland-coverage/farmland-coverage.ipynb) | This notebook demonstrates how to use a VisualPrompting model to analyze the area coverage of different types of land or structures on satellite images. | Jupyter Notebook |

## Install the library

```bash
pip install landingai
```

## Quick Start

**Prerequisites**

This library needs to communicate to the LandingAI platform for various functionalities (e.g. the `Predictor` API, it calls the HTTP endpoint of your deployed model for prediction results). Thus, you need to have below information at hand before using those functionalities:

1. LandingAI user **API credentials** (API key and API secret). See [here](https://support.landing.ai/docs/api-key-and-api-secret?highlight=api%20key) for how to get it.
2. The **Endpoint ID** of your deployed model on CloudInference LandingAI. See [here](https://support.landing.ai/landinglens/docs/cloud-deployment) for how to get it.

**Run inference** using your deployed inference endpoint at LandingAI:

- Install the library with the above command.
- Create a `Predictor` with your inference endpoint id, landing API key and secret.
- Call `predict()` with an image (in numpy array format).

```python
from landingai.predict import Predictor
# Find your API key and secrets
endpoint_id = "FILL_YOUR_INFERENCE_ENDPOINT_ID"
api_key = "FILL_YOUR_API_KEY"
api_secret = "FILL_YOUR_API_SECRET"
# Load your image
image = ...
# Run inference
predictor = Predictor(endpoint_id, api_key, api_secret)
predictions = predictor.predict(image)
```

**Visualize your inference results** by overlaying the predictions on the input image and save it on disk:

```python
from landingai.visualize import overlay_predictions
# continue the above example
predictions = predictor.predict(image)
image_with_preds = overlay_predictions(predictions, image)
image_with_preds.save("image.jpg")
```

**Storing API credentials**

There are three ways to configure your user API credentials:

1. Pass them as function parameters.

2. Set them as environment variables, e.g. `export LANDINGAI_API_KEY=...`, `export LANDINGAI_API_SECRET=...`

3. Store them in an `.env` file under your project root directory. E.g. below is an example credential data in `.env` file.

   ```
   LANDINGAI_API_KEY=v7b0hdyfj6271xy2o9lmiwkkcb12345
   LANDINGAI_API_SECRET=ao6yjcju7q1e6u0udgwrgknhrx6m4n1o48z81jy6huc059gne047l4fq312345
   ```

The above ordering also indicates the priority of the credential loading order.

## Documentations

#### 1. [LANDING AI Python Library API Reference](https://landing-ai.github.io/landingai-python/landingai.html)

#### 2. LANDING AI Python Library User Guide (coming soon)

#### 3. [LANDING AI Platform Suport Center](https://support.landing.ai/)

#### 4. [Quick LandingLens Video Walk-Through](https://support.landing.ai/docs/landinglens-workflow) 

## Running examples locally

All the examples in this repo can be run locally.

Here is an example to show you how to run the `rtsp-capture` example locally in a shell environment:

1. Clone the repo to local: `git clone https://github.com/landing-ai/landingai-python.git`
2. Install the library: `poetry install --with examples` (NOTE: see below for how to install `poetry`)
3. Activate the virtual environment: `poetry shell`
4. Run: `python landingai-python/examples/capture-service/run.py`

## Building the `landingai` library locally (for contributors)

Most of the time you won't need to build the library since it is included on this repository and also published to pypi.

But if you want to contribute to the repo, you can follow the below steps.

### Prerequisite - Install poetry

> `landingai` uses `Poetry` for packaging and dependency management. If you want to build it from source, you have to install Poetry first. Please follow
[the official guide](https://python-poetry.org/docs/#installation) to see all possible options.

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

### Install all the dependencies

```bash
poetry install --all-extras
```

### Run tests

```bash
poetry run pytest tests/
```

### Activate the virtualenv

```bash
poetry shell
```
