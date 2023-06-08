## Introduction

This example focuses on how to continuously run inference on images extracted from streaming video. This application shows how to use a Segmentation model from LandingLens to detect sky and clouds in images extracted from a streaming RTSP video camera feed. A traffic camera is used to capture images.  

## Run the Example

### Prerequisites

Before starting, install the Landing AI Python library. For more information, see the top-level `README.md`.

## Run the Example Source and Inference Model

To launch the program, run this command:

```bash
python examples/capture-service/run.py 
```

The program captures frames from the video feed every few seconds, and then runs inference on those images. After inference is complete, a pop-up window appears, showing the captured image with the model's predictions overlaid on it.

> Note: The endpoints for Free Trial LandingLens accounts have a limit of 20 inferences/minute. Do not call inference more than that rate. You can change the inference frequency by configuring a constant in `_CAPTURE_INTERVAL` in the `run.py` file.

## Customize the Example

1. Set up a camera that exposes an RTSP URL to your network (your local intranet). If you're not sure if the RTSP URL is working, learn how to test it in this [article](https://support.ipconfigure.com/hc/en-us/articles/115005588503-Using-VLC-to-test-camera-stream).
2. Train a model in LandingLens, and deploy it to an endpoint via [Cloud Deployment](https://support.landing.ai/landinglens/docs/cloud-deployment).
3. Get the `endpoint id`, `api key` and `api secret` from LandingLens.
4. Open the file `examples/capture-service/run.py`, and update the following with your information: `api_key`, `api_secret`, `endpoint_id` and `stream_url`. 


## Collect Images to Train Your Model

You need to collect images and train your model with those images before you can run inference. You can use the same script to collect training images by setting `capture_frame` to `True` and `inference_mode` to `False`. The default parameters of the `stream()` function are:

```
def stream(capture_frame=False, inference_mode=True):
    ...
```
