<p align="center">
  <img width="100" height="100" src="https://github.com/landing-ai/landingai-python/raw/main/assets/avi-logo.png">
</p>

# LandingLens Python SDK
The LandingLens Python SDK contains the LandingLens development library and examples that show how to integrate your app with LandingLens in a variety of scenarios. The examples cover different model types, image acquisition sources, and post-procesing techniques. 

We've provided some examples in Jupyter Notebooks to focus on ease of use, and some examples in Python apps to provide a more robust and complete experience.

<!-- Generated using https://www.tablesgenerator.com/markdown_tables -->

| Example | Description | Type |
|---|---|---|
| [Poker Card Suit Identification](https://github.com/landing-ai/landingai-python/blob/main/examples/webcam-collab-notebook/webcam-collab-notebook.ipynb) | This notebook shows how to use an object detection model from LandingLens to detect suits on playing cards. A webcam is used to take photos of playing cards. | Jupyter Notebook [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/landing-ai/landingai-python/blob/main/examples/webcam-collab-notebook/webcam-collab-notebook.ipynb)|
| [Door Monitoring for Home Automation](https://github.com/landing-ai/landingai-python/blob/main/examples/rtsp-capture-notebook/rtsp-capture.ipynb) | This notebook shows how to use an object detection model from LandingLens to detect whether a door is open or closed. An RTSP camera is used to acquire images. | Jupyter Notebook |
| [Streaming Video](https://github.com/landing-ai/landingai-python/tree/main/examples/capture-service) | This application shows how to continuously run inference on images extracted from a streaming RTSP video camera feed. | Python application |
| [Satellite Images and Post-Processing](https://github.com/landing-ai/landingai-python/tree/main/examples/post-processings/farmland-coverage/farmland-coverage.ipynb) | This notebook shows how to use a Visual Prompting model from LandingLens to identify different objects in satellite images. The notebook includes post-processing scripts that calculate the percentage of ground cover that each object takes up. | Jupyter Notebook |

## Install the Library

```bash
pip install landingai
```

## Quick Start

### Prerequisites

This library needs to communicate with the LandingLens platform to perform certain functions. (For example, the `Predictor` API calls the HTTP endpoint of your deployed model). To enable communication with LandingLens, you will need the following information:

1. The **Endpoint ID** of your deployed model in LandingLens. You can find this on the Deploy page in LandingLens.
2. The **API Key** and **API Secret** for the LandingLens organization that has the model you want to deploy. To learn how to generate these credentials, go [here](https://support.landing.ai/docs/api-key-and-api-secret).


### Run Inference
Run inference using the endpoint you created in LandingLens:

- Install the Python library.
- Create a `Predictor` fucntion with your Endpoint ID, API Key, and API Secret.
- Call the `predict()` function with an image (using the NumPy array format).

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

See a **working example** in [here](https://github.com/landing-ai/landingai-python/blob/main/tests/landingai/test_predict.py).

### Visualize and Save Predictions
Visualize your inference results by overlaying the predictions on the input image and saving the updated image:

```python
from landingai.visualize import overlay_predictions
# continue the above example
predictions = predictor.predict(image)
image_with_preds = overlay_predictions(predictions, image)
image_with_preds.save("image.jpg")
```

### Store API Credentials

Here are the three ways to configure your API Key and API Secret, ordered by the priority in which they are loaded:

1. Pass them as function parameters.

2. Set them as environment variables. For example: `export LANDINGAI_API_KEY=...`, `export LANDINGAI_API_SECRET=...`.

3. Store them in an `.env` file under your project root directory. For example, here is a set of credentials in an `.env` file:

```
   LANDINGAI_API_KEY=v7b0hdyfj6271xy2o9lmiwkkcb12345
   LANDINGAI_API_SECRET=ao6yjcju7q1e6u0udgwrgknhrx6m4n1o48z81jy6huc059gne047l4fq312345
```

## Run Examples Locally

All the examples in this repo can be run locally.

To give you some guidance, here's how you can run the `rtsp-capture` example locally in a shell environment:

1. Clone the repo to local: `git clone https://github.com/landing-ai/landingai-python.git`
2. Install the library: `poetry install --with examples` (Note: See below for how to install `poetry`)
3. Activate the virtual environment: `poetry shell`
4. Run: `python landingai-python/examples/capture-service/run.py`

## Build the `landingai` Library Locally (for Contributors)

Most of the time you won't need to build the library because it is included in this repository and also published to PyPI. But if you'd like to contribute to this repository, follow the instructions below.

### Prerequisite - Install Poetry

> `landingai` uses `Poetry` for packaging and dependency management. If you want to build it from source, you have to install Poetry first. To see all possible options, go to [the official guide](https://python-poetry.org/docs/#installation).

For Linux, macOS, and Windows (WSL):

```
curl -sSL https://install.python-poetry.org | python3 -
```

Note: You can switch to use a different Python version by specifying the Python version:

```
curl -sSL https://install.python-poetry.org | python3.10 -
```

Or run the following command after you install Poetry:

```
poetry env use 3.10
```

### Install All Dependencies

```bash
poetry install --all-extras
```

### Run Tests

```bash
poetry run pytest tests/
```

### Activate the virtualenv

```bash
poetry shell
```

## Documentation

-  [Landing AI Python Library API Reference](https://landing-ai.github.io/landingai-python/landingai.html)

-  [Landing AI Suport Center](https://support.landing.ai/)

-  [LandingLens Walk-Through Video](https://www.youtube.com/watch?v=779kvo2dxb4)

-  Landing AI Python Library User Guide (coming soon)
