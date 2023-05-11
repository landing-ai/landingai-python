<p align="center">
  <img width="200" height="200" src="assets/avi-logo.png">
</p>

# LandingLens code sample repository
This repository contains LandingLens development library and running examples showing how to integrate LandingLens on a variety of scenarios.

<!-- Generated using https://www.tablesgenerator.com/markdown_tables -->
| example | description | language |
|---|---|---|
| [Door monitoring for home automation](examples/rtsp-capture-notebook/rtsp-capture.ipynb) | This notebook uses an object detection model to determine whether a door is open or closed. the notebook can acquire images directly from an RTSP camera | Jupyter Notebook [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/landing-ai/landingai-python-v1/blob/main/examples/rtsp-capture-notebook/rtsp-capture.ipynb)|
| [Cosmic ray detector](examples/capture-service/cosmic-rays.py) | This application shows how to do continuous acquisition from an image sensor using RTSP. Images are analyzed using a segmentation model to detect cosmic rays | Python application |

## Running the examples
all the examples can be run locally or using 
```bash
git clone
pip install
```

## Install the LandingLens library locally

Most of the time you won't need to build the library since it is included on this repository and also published to pypi.

But if you want to contribute to the repo, you can follow the below steps:

1. **Install poetry**

| See more from the [official doc](https://python-poetry.org/docs/#installation).

For Linux, macOS, Windows (WSL):

```
curl -sSL https://install.python-poetry.org | python3 -
```

NOTE: if you can switch to use a different Python version by specifying the python version:

```
curl -sSL https://install.python-poetry.org | python3.10 -
```

or run below command after you have installed poetry:

```
poetry env use 3.10
```

2. **Install all the dependencies**

```bash
poetry install
```
