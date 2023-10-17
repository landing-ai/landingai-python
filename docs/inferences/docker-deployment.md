Running inferences with the standard `landingai.predict.Predictor` will send your image to LandingLens cloud, which is ideal if you don't want to worry about backend scalability, hardware provisioning, availability, etc. But this also adds some networking overhead that might limit how many inferences per second you can run.

If you need to run several inferences per second, and you have your own cloud service or local machine, you might want to run inference using your own resources. For that, we provide **[Docker deployment](https://support.landing.ai/docs/docker-deploy)**, a Docker image with your LandingLens trained model embeded that you can run anywhere.


???+ note

    You can get more details on how to set up and run the Docker deployment container locally or in your own cloud service in our [Support Center](https://support.landing.ai/docs/docker-deploy).

    Once you go through the Support Center guide, you will have the model running in a container, accessible in a specific host and port. The example below refers to these as `localhost` and `8000`, respectively.


Once you have your Docker deployment container running, you can run inference using the `landingai.predict.EdgePredictor` class. For example:

```py
from landingai.pipeline.image_source import Webcam
from landingai.predict import EdgePredictor

predictor = EdgePredictor(host="localhost", port=8000) # (1)!
with Webcam(fps=15) as webcam: # (2)!
    for frame in webcam:
        frame.run_predict(predictor=predictor) # (3)!
        frame.overlay_predictions()
        if "coffee-mug" in frame.predictions:  # (4)!
            frame.save_image(
                "/tmp/latest-webcam-image.png",
                include_predictions=True
            )
```

1. Create an `EdgePredictor` object, specifying the host and port where your Docker container is running.
2. Capture images from the webcam at 15 frames per second, closing the camera at the end of the `with` block.
3. Run inference on the frame using the Docker container predictor.
4. If the model detects a "coffee-mug" object in the frame, save the image to a file, including the predictions as an overlay.

The `EdgePredictor` class is a subclass of `Predictor`, so you can use it in the same way as the standard `Predictor` class. The only difference is that the `EdgePredictor` will send the image to your Docker container instead of sending it to LandingLens cloud.

The time it takes to run the inference will vary according to the hardware where the Docker container is running (if you set it up to run on a GPU, for example, it will probably yield faster predictions).

Check out the [Support Center](https://support.landing.ai/docs/docker-deploy) for more information on how to get a deployment license, run the Docker deployment with a a GPU, and more.