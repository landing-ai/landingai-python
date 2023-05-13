## Introduction

This example shows how to connect to a camera (that supports RTSP protocol), and continuously run inference on captured images.

### How to run the example

**Prerequisite**: 

1. you have installed this library correctly. See the top-level `README.md` for more details.
2. you have a working camera that exposes a RTSP url to your network, e.g. your local intranet. (Not sure if the RTSP url is working? check out this [article](https://support.ipconfigure.com/hc/en-us/articles/115005588503-Using-VLC-to-test-camera-stream) to test it out)
3. you have trained a Landing AI model, and deployed such a model to a CloudInference endpoint.
4. you have `endpoint id`, `api key` and `api secret` from the Landing AI platform.

Action required:

Open the file `examples/capture-service/run.py`, and change the `api_key`, `api_secret`, `endpoint_id` and `stream_url` to yours. (Tip: search for "TODO" in the file)

#### Run inference

Launch the program by

```bash
python examples/capture-service/run.py 
```

If successful, the program will capture a frame every 3 seconds, and make inference against every captured frame. You should see a Window pop up and show you the captured image with predictions overlaid on it.
The image with overlaid predictions will also be saved on disk under `examples/output/capture-service`.

> NOTE: the CloudInference endpoint of a free trial account has a rate limit of 20 inferences/minute. So don't call inference too frequently. The frequency can be configured by a constant in the `run.py` file, see `_CAPTURE_FREQUENCY_SECONDS`.

#### Collect training images

You need to collect training images and train a model before you can run inferences.
You can use the same script to collect training images by setting `capture_frame` to `True` and `inference_mode` to `False`. The default parameters of the `stream()` function is:

```
def stream(capture_frame=False, inference_mode=True):
    ...
```
