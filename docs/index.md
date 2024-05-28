<p align="center">
  <img width="100" height="100" src="https://github.com/landing-ai/landingai-python/raw/main/assets/avi-logo.png">
</p>

# Welcome to the LandingAI Python Library Documentation

The LandingAI Python Library is a set of tools to help you build computer vision applications. While some of the functionality is specific to [LandingLens](https://app.landing.ai/), the computer vision platform from LandingAI, other features can be used for managing images in general.

The library includes features to acquire, process, and detect objects in your images and videos, with the least amount of code as possible.

## Quick Start

### Install
First, install the LandingAI Python library:

```bash
pip install landingai~=0.3.0
```


### Acquire Your First Images

After installing the LandingAI Python library, you can start acquiring images from one of many image sources. 

For example, from a single image file:

```py
from landingai.pipeline.frameset import Frame

frame = Frame.from_image("/path/to/your/image.jpg") # (1)!
frame.resize(width=512, height=512) # (2)!
frame.save_image("/tmp/resized-image.png") # (3)!
```

1. We support several image file types. See the full list [here](https://support.landing.ai/docs/upload-images).
2. Resize the frame to 512x512p.
3. Save the resized image to `/tmp/resized-image.png`.


You can also extract frames from your webcam. For example:

```py
from landingai.pipeline.image_source import Webcam

with Webcam(fps=0.5) as webcam:  # (1)!
    for frame in webcam:
        frame.resize(width=512, height=512) # (2)!
        frame.save_image("/tmp/webcam-image.png") # (3)!
```

1. Capture images from the webcam at 0.5 frames per second (1 frame every 2 seconds), closing the camera at the end of the `with` block.
2. Resize the frame to 512x512p.
3. Save the images as `/tmp/webcam-image.png`.


To learn how to acquire images from more sources, go to [Image Acquisition](image-acquisition/image-acquisition.md).


### Run Inference

If you have deployed a computer vision model in LandingLens, you can use this library to send images to that model for inference.

For example, let's say we've created and deployed a model in LandingLens that detects coffee mugs. Now, we'll use the code below to extract images (frames) from a webcam and run inference on those images.

???+ note

    If you don't have a LandingLens account, create one [here](https://app.landing.ai/). Learn how to use LandingLens from our [Support Center]([https://support.landing.ai/docs/landinglens-workflow](https://support.landing.ai/landinglens/en)) and [Video Tutorial Library](https://support.landing.ai/docs/landinglens-workflow-2). Need help with specific use cases? Post your questions in our [Community](https://community.landing.ai/home).


???+ note
    If you are running LandingLens as a Snowflake Native App, see the [Snowflake Native App](inferences/snowflake-native-app.md) section for more information.


```py
from landingai.pipeline.image_source import Webcam
from landingai.predict import Predictor

predictor = Predictor(  # (1)!
    endpoint_id="abcdef01-abcd-abcd-abcd-01234567890", # (2)!
    api_key="land_sk_xxxxxx", # (3)!
)
with Webcam(fps=0.5) as webcam:
    for frame in webcam:
        frame.resize(width=512) # (4)!
        frame.run_predict(predictor=predictor) # (5)!
        frame.overlay_predictions()
        if "coffee-mug" in frame.predictions:  # (6)!
            frame.save_image("/tmp/latest-webcam-image.png", include_predictions=True) # (7)!
```

1. Creates a LandingLens predictor object.
2. Set the `endpoint_id` to the one from your deployed model in LandingLens: <br/>![How to get endpoint ID](images/copy-endpoint-id.png "How to get endpoint ID").
3. Set the `api_key` to the one from your LandingLens organization: <br/> ![How to get the API key](images/menu-api-key.png "How to get the API key").
4. Resize the image to `width=512`, keeping the aspect ratio. This is useful to save some bandwidth when sending the image to LandingLens for inference.
5. Runs inference in the resized frame, and adds the predictions to the `Frame`.
6. If the model predicts that an object with a class named `coffee-mug` is found...
7. Then save the image with the predictions overlaid on the image.

That's it! Now, with just a few lines of code, you can detect coffee mugs in front of your webcam.

Now, learn about the other ways to [acquire images](image-acquisition/image-acquisition.md), and how to process your images and [run inference on them](inferences/getting-started.md). For inspiration, check out our [Examples](examples.md).
