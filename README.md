<p align="center">
  <img width="200" height="200" src="https://github.com/landing-ai/landingai-python-v1/raw/main/assets/avi-logo.png">
</p>

# LandingLens code sample repository
This repository contains LandingLens development library and running examples showing how to integrate LandingLens on a variety of scenarios. All the examples show different ways to acquire images from multiple sources and techniques to process the results. Jupyter notebooks focus on ease of use while Python apps include more robust and complete examples.

<!-- Generated using https://www.tablesgenerator.com/markdown_tables -->
| example | description | language |
|---|---|---|
| [Company logo identification](https://github.com/landing-ai/landingai-python-v1/blob/main/examples/webcam-collab-notebook/webcam-collab-notebook.ipynb) | This notebook can run directly in Google collab using the web browser camera to detect Landing AI logo | Jupyter Notebook [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/landing-ai/landingai-python-v1/blob/main/examples/webcam-collab-notebook/webcam-collab-notebook.ipynb)|
| [Door monitoring for home automation](https://github.com/landing-ai/landingai-python-v1/blob/main/examples/rtsp-capture-notebook/rtsp-capture.ipynb) | This notebook uses an object detection model to determine whether a door is open or closed. The notebook can acquire images directly from an RTSP camera | Jupyter Notebook |
| [Streaming capture service (WIP)](examples/capture-service/run.py) | This application shows how to do continuous acquisition from an image sensor using RTSP. | Python application |

## Install the library

```bash
pip install landingai
```

## Running examples locally

All the examples in this repo can be run locally. 

Here is an example to show you how to run the `rtsp-capture` example locally in a shell environment:

NOTE: it's recommended to create a new Python virtual environment first.

1. Clone the repo to local: `git clone https://github.com/landing-ai/landingai-python-v1.git`
2. Install the library: `pip install landingai`
3. Run: `python landingai-python-v1/examples/rtsp-capture/run.py`

## Building the LandingLens library locally (for developers and contributors)

Most of the time you won't need to build the library since it is included on this repository and also published to pypi.

But if you want to contribute to the repo, you can follow the below steps:

### Install poetry

> See more from the [official doc](https://python-poetry.org/docs/#installation).

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
poetry install --with test
```

### Run tests

```bash
poetry run pytest tests/
```
