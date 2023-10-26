![ci_status](https://github.com/landing-ai/landingai-python/actions/workflows/ci_cd.yml/badge.svg)
[![PyPI version](https://badge.fury.io/py/landingai.svg?)](https://badge.fury.io/py/landingai)
![version](https://img.shields.io/pypi/pyversions/landingai)
![license](https://img.shields.io/github/license/landing-ai/landingai-python)
[![downloads](https://static.pepy.tech/badge/landingai/month)](https://pepy.tech/project/landingai)

<br>

<p align="center">
  <img width="100" height="100" src="https://github.com/landing-ai/landingai-python/raw/main/assets/avi-logo.png">
</p>

# LandingLens Python SDK
The LandingLens Python SDK contains the LandingLens development library and examples that show how to integrate your app with LandingLens in a variety of scenarios. The examples cover different model types, image acquisition sources, and post-procesing techniques.

## Documentation

-  [Landing AI Python SDK Docs](https://landing-ai.github.io/landingai-python/)
-  [Landing AI Support Center](https://support.landing.ai/)
-  [LandingLens Walk-Through Video](https://www.youtube.com/watch?v=779kvo2dxb4)


## Quick start

### Install
First, install the Landing AI Python library:

```bash
pip install landingai
```


### Acquire Your First Images

After installing the Landing AI Python library, you can start acquiring images from one of many image sources.

For example, from a single image file:

```py
from landingai.pipeline.frameset import Frame

frame = Frame.from_image("/path/to/your/image.jpg")
frame.resize(width=512, height=512)
frame.save_image("/tmp/resized-image.png")
```

You can also extract frames from your webcam. For example:

```py
from landingai.pipeline.image_source import Webcam

with Webcam(fps=0.5) as webcam:
    for frame in webcam:
        frame.resize(width=512, height=512)
        frame.save_image("/tmp/webcam-image.png")
```


To learn how to acquire images from more sources, go to [Image Acquisition](https://landing-ai.github.io/landingai-python/image-acquisition/image-acquisition/).


### Run Inference

If you have deployed a computer vision model in LandingLens, you can use this library to send images to that model for inference.

For example, let's say we've created and deployed a model in LandingLens that detects coffee mugs. Now, we'll use the code below to extract images (frames) from a webcam and run inference on those images.

> [!NOTE]
> If you don't have a LandingLens account, create one [here](https://app.landing.ai/). You will need to get an "endpoint ID" and "API key" from LandingLens in order to run inferences. Check our [Running Inferences / Getting Started](https://landing-ai.github.io/landingai-python/inferences/getting-started/).

> [!NOTE]
> Learn how to use LandingLens from our [Support Center]([https://support.landing.ai/docs/landinglens-workflow](https://support.landing.ai/landinglens/en)) and [Video Tutorial Library](https://support.landing.ai/docs/landinglens-workflow-2).
> Need help with specific use cases? Post your questions in our [Community](https://community.landing.ai/home).


```py
from landingai.pipeline.image_source import Webcam
from landingai.predict import Predictor

predictor = Predictor(
    endpoint_id="abcdef01-abcd-abcd-abcd-01234567890",
    api_key="land_sk_xxxxxx",
)
with Webcam(fps=0.5) as webcam:
    for frame in webcam:
        frame.resize(width=512)
        frame.run_predict(predictor=predictor)
        frame.overlay_predictions()
        if "coffee-mug" in frame.predictions:
            frame.save_image("/tmp/latest-webcam-image.png", include_predictions=True)
```


## Examples

We've provided some examples in Jupyter Notebooks to focus on ease of use, and some examples in Python apps to provide a more robust and complete experience.

<!-- Generated using https://www.tablesgenerator.com/markdown_tables -->

| Example | Description | Type |
|---|---|---|
| [Poker Card Suit Identification](https://github.com/landing-ai/landingai-python/blob/main/examples/webcam-collab-notebook/webcam-collab-notebook.ipynb) | This notebook shows how to use an object detection model from LandingLens to detect suits on playing cards. A webcam is used to take photos of playing cards. | Jupyter Notebook [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/landing-ai/landingai-python/blob/main/examples/webcam-collab-notebook/webcam-collab-notebook.ipynb)|
| [Door Monitoring for Home Automation](https://github.com/landing-ai/landingai-python/blob/main/examples/rtsp-capture-notebook/rtsp-capture.ipynb) | This notebook shows how to use an object detection model from LandingLens to detect whether a door is open or closed. An RTSP camera is used to acquire images. | Jupyter Notebook [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/landing-ai/landingai-python/blob/main/examples/rtsp-capture-notebook/rtsp-capture.ipynb) |
| [Satellite Images and Post-Processing](https://github.com/landing-ai/landingai-python/tree/main/examples/post-processings/farmland-coverage/farmland-coverage.ipynb) | This notebook shows how to use a Visual Prompting model from LandingLens to identify different objects in satellite images. The notebook includes post-processing scripts that calculate the percentage of ground cover that each object takes up. | Jupyter Notebook [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/landing-ai/landingai-python/blob/main/examples/post-processings/farmland-coverage/farmland-coverage.ipynb) |
| [License Plate Detection and Recognition](https://github.com/landing-ai/landingai-python/tree/main/examples/license-plate-ocr-notebook/license_plate_ocr.ipynb) | This notebook shows how to extract frames from a video file and use a object detection model and OCR from LandingLens to identify and recognize different license plates. | Jupyter Notebook [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/landing-ai/landingai-python/blob/main/examples/license-plate-ocr-notebook/license_plate_ocr.ipynb) |
| [Streaming Video](https://github.com/landing-ai/landingai-python/tree/main/examples/capture-service) | This application shows how to continuously run inference on images extracted from a streaming RTSP video camera feed. | Python application |


## Run Examples Locally

All the examples in this repo can be run locally.

To give you some guidance, here's how you can run the [`rtsp-capture`](https://github.com/landing-ai/landingai-python/tree/main/examples/capture-service) example locally in a shell environment:

1. Clone the repo to local: `git clone https://github.com/landing-ai/landingai-python.git`
2. Install the library: `poetry install --with examples` (See the [poetry docs](https://python-poetry.org/docs/#installation) for how to install `poetry`)
3. Activate the virtual environment: `poetry shell`
4. Run: `python landingai-python/examples/capture-service/run.py`
