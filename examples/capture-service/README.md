## Introduction

This example shows how to connect to a networked camera (that supports RTSP protocol), and continuously run inference on captured images.

### How to run the example

**Prerequisites**: 

1. you have installed this library correctly. See the top-level `README.md` for more details.

#### Running the example source and inference model

Launch the program by

```bash
python examples/capture-service/run.py 
```

If successful, the program will capture frames every couple of seconds, and make inference against. You should see a Window pop up and show you the captured image with predictions overlaid on it. The example source is a traffic camera paired with a model that detect sky and clouds

> NOTE: the CloudInference endpoint of a free trial account has a rate limit of 20 inferences/minute. So don't call inference too frequently. The frequency can be configured by a constant in the `run.py` file, see `_CAPTURE_INTERVAL`.

#### Customizing the example

1. you need a working camera that exposes a RTSP url to your network, e.g. your local intranet. (Not sure if the RTSP url is working? check out this [article](https://support.ipconfigure.com/hc/en-us/articles/115005588503-Using-VLC-to-test-camera-stream) to test it out)
2. you have trained a Landing AI model, and deployed such a model to a CloudInference endpoint.
3. you have `endpoint id`, `api key` and `api secret` from the Landing AI platform.
4. Open the file `examples/capture-service/run.py`, and change the `api_key`, `api_secret`, `endpoint_id` and `stream_url` to yours. 


#### Collect training images

You need to collect training images and train a model before you can run inferences.
You can use the same script to collect training images by setting `capture_frame` to `True` and `inference_mode` to `False`. The default parameters of the `stream()` function is:

```
def stream(capture_frame=False, inference_mode=True):
    ...
```
