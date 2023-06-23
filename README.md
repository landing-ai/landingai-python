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
- Create a `Predictor` class with your Endpoint ID, API Key, and API Secret.
- Load your image into a NumPy array (below the image is "image.png")
- Call the `predict()` function with an image (using the NumPy array format).

```python
imoprt numpy as np
from PIL import Image
from landingai.predict import Predictor
# Find your API key and secrets
endpoint_id = "FILL_YOUR_INFERENCE_ENDPOINT_ID"
api_key = "FILL_YOUR_API_KEY"
api_secret = "FILL_YOUR_API_SECRET"
# Load your image
image = np.asarray(Image.open("image.png"))
# Run inference
predictor = Predictor(endpoint_id, api_key, api_secret)
predictions = predictor.predict(image)
```

See a **working example** in [here](https://github.com/landing-ai/landingai-python/blob/main/tests/integration/landingai/test_predict_e2e.py).

### Visualize and Save Predictions
Visualize your inference results by overlaying the predictions on the input image and saving the updated image:

```python
from landingai.visualize import overlay_predictions
# continue the above example
predictions = predictor.predict(image)
image_with_preds = overlay_predictions(predictions, image)
image_with_preds.save("image.jpg")
```
### Putting together a vision pipeline

All the modules shown above and others can be chained together using the `landingai.vision_pipeline` abstraction. At its core, a pipeline is a sequence of chained calls that operate on a `landingai.vision_pipeline.FrameSet`.

The following example shows how the previous sections come together on a pipeline. For more details check the [*Vision Pipelines user guide*](#vision-pipelines) 
```python
cloud_sky_model = Predictor("FILL_YOUR_INFERENCE_ENDPOINT_ID"
                            , "FILL_YOUR_API_KEY"
                            , "FILL_YOUR_API_SECRET") 

Camera = NetworkedCamera(stream_url)
for frame in Camera:
    (
        frame.downsize(width=1024)
        .run_predict(predictor=cloud_sky_model)
        .overlay_predictions()
        .show_image()
        .save_image(filename_prefix="./capture")
    )
```

## Run Examples Locally

All the examples in this repo can be run locally.

To give you some guidance, here's how you can run the `rtsp-capture` example locally in a shell environment:

1. Clone the repo to local: `git clone https://github.com/landing-ai/landingai-python.git`
2. Install the library: `poetry install --with examples` (Note: See [Developer Guide](https://landing-ai.github.io/landingai-python/landingai.html#developer-guide) for how to install `poetry`)
3. Activate the virtual environment: `poetry shell`
4. Run: `python landingai-python/examples/capture-service/run.py`

## Documentation

-  [Landing AI Python Library API Reference](https://landing-ai.github.io/landingai-python/landingai.html)
-  [Landing AI Python Library User Guide](https://landing-ai.github.io/landingai-python/landingai.html#user-guide)
-  [Landing AI Python Library Developer Guide](https://landing-ai.github.io/landingai-python/landingai.html#developer-guide)
-  [Landing AI Support Center](https://support.landing.ai/)
-  [LandingLens Walk-Through Video](https://www.youtube.com/watch?v=779kvo2dxb4)
