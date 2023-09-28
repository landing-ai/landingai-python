We've provided some examples in Jupyter Notebooks to focus on ease of use, and some examples in Python apps to provide a more robust and complete experience.

If you have a cool app using Landing.ai SDK and you would like to have it featured here, please [let us know](https://github.com/landing-ai/landingai-python/issues/new).

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

To give you some guidance, here's how you can run the `rtsp-capture` example locally in a shell environment:

1. Clone the repo to local: `git clone https://github.com/landing-ai/landingai-python.git`
2. Install the library: `poetry install --with examples` (See the [Developer Guide](https://landing-ai.github.io/landingai-python/landingai.html#developer-guide) for how to install `poetry`)
3. Activate the virtual environment: `poetry shell`
4. Run: `python landingai-python/examples/capture-service/run.py`
