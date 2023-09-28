# Welcome to Landing.ai SDK documentation

Landing.ai SDK is a set of tools to help you build computer vision applications, including
integration with [LandingLens](https://app.landing.ai/).

The SDK includes helper features to acquire, process and detect things in your images and videos, with the minimum lines of code possible.

## Quick start

First, install the SDK:

```bash
pip install landingai
```

After that, you can start acquiring images from one of our image sources.

For example, from your webcam:

```py
from landingai.pipeline.image_source import Webcam

webcam = Webcam(fps=0.5)  # (1)!
for frame in webcam:
    resized_frame = frame.resize(width=512, height=512) # (2)!
    resized_frame.save_image("/tmp/webcam-image-") # (3)!
```

1. Capture images from the webcam at 0.5 frames per second (1 frame every 2 seconds)
2. Resize the frame to 512x512 pixels
3. Save the images as `/tmp/webcam-image-<timestamp>.png`

Nice. Now, if you already have a LandingLens account and a trained model, you
can start detecting things in your images.

Let's use the images from our webcam and our coffee mug detector model, already deployed in LandingLens:

???+ note

    If you still don't have a LandingLens account, create an account [here](https://app.landing.ai/)
    and deploy your first model.
    Visit the [Support Center](https://support.landing.ai/docs/landinglens-workflow) for a walkthrough on how to deploy your first model, and how to get [your API key](https://support.landing.ai/docs/api-key).

```py
from landingai.pipeline.image_source import Webcam

predictor = Predictor(  # (1)!
    endpoint_id="abcdef01-abcd-abcd-abcd-01234567890", # (2)!
    api_key="land_sk_xxxxxx", # (3)!
)
webcam = Webcam(fps=0.5)
for frame in webcam:
    resized_frame = frame.resize(width=512, height=512)
    frame_with_predictions = resized_frame.run_predict(predictor=predictor) # (4)!
    if "coffee-mug" in frame_with_predictions.predictions:  # (5)!
        resized_frame.save_image("/tmp/webcam-image-", image_src="overlay") # (6)!
```

1. asdfasdfasdf
2. fffff
3. asdfasdf
4. asdf
5. asdf
6. asdf

That's it. Now, with just a few lines of code, you detect coffee mugs passing by your webcam.

You can now check the other image sources available, and other helpers to process your images and the predictions you do on them.