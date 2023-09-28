## Vision Pipelines

### Image acquisition
Pipelines can simplify complex vision tasks by breaking them into a sequence of operations that are applied to every image. Images are modeled as `landingai.pipeline.frameset.Frame` and sequences of images are modeled as `landingai.pipeline.frameset.FrameSet`. Most image sources will produce a `FrameSet` even when it contains a single image.

For example, a `landingai.pipeline.image_source.NetworkedCamera` can connect to a live video source and expose it as `FrameSet` iterator. This is convenient as it allows subsequent stages of the pipeline to introduce new derived frames as part of the processing.

Other examples of data acquisition classes are `landingai.pipeline.image_source.Webcam`, to collect images directly from the webcam of the current device; and `landingai.pipeline.image_source.Screenshot`, that takes screenshots of the current device.

### Running predictions

You can use data pipelines to run predictions quite easily, using the `landingai.predict.Predictor` class. You just need to get from the platform the endpoint_id to where your model was deployed, and your API key:

```python
predictor = Predictor(endpoint_id="abcde-1234-xxxx", api_key="land_sk_xxxx")

# Get images from Webcam at a 1 FPS rate, run predictions on it and save results.
with webcam as Webcam(fps=1):
    for frame in webcam:
        frame
            .downsize(width=512)
            .run_predict(predictor=predictor)
            .save_image(f"/tmp/detected-object", image_src="overlay")
```

You can also check the prediction result using some of the helper methods:

```python
predictor = Predictor(endpoint_id="abcde-1234-xxxx", api_key="land_sk_xxxx")
# Get images from Webcam at a 1 FPS rate, run predictions on it and check if
# in the prediction we find "coffee-mug", a class created in LandingLens platform:
with webcam as Webcam(fps=1):
    for frame in webcam:
        frame = frame
            .downsize(width=512)
            .run_predict(predictor=predictor)
        if "coffee-mug" in frame.predictions:
            print(f"Found {len(frame.predictions)} coffee mugs in the image")
```

FrameSet predictions has also other methods to help filter predicted classes:

```python
# Returns only the predictions with a confidence score above 0.9
sure_predictions = frame.predictions.filter_threshold(0.9)

# Returns only predictions for the "teapot" class
teapot_predictions = frame.predictions.filter_label("teapot")
```

As an other example, if we are detecting faces from a live stream, we may want to first use an object detection model to identify the regions of interest and then break the initial `Frame` into multiple frames (one per face). Subsequent stages of the pipeline may apply other models to individual faces.

The following example shows a basic pipeline that applies several processing layers starting from a single image. In this case after running inference (i.e. `run_predict`), the `overlay_predictions` function creates a copy of the original image (i.e. `frs[0].image`) and populates `frs[0].other_images["overlay"]` with the results.

```python
frs= FrameSet.from_image("sample_images/1196.png")
frs.run_predict(predictor=...)
.overlay_predictions()
.show_image()
.show_image(image_src="overlay")

```

For more details on the operations supported on pipelines, go to `landingai.pipeline.frameset.FrameSet`.
