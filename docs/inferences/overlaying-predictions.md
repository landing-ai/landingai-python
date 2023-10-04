Apart from counting and checking the predictions, we can also overlay the predictions on the original image. This is useful to see how well the model is performing. We can do this by using the `Frame.save_image` function. This function merges the original `Frame` image, the predictions, and the labels. It then overlays the predictions on top of the original image.

```python
from landingai.pipeline.image_source import Webcam
from landingai.predict import Predictor

predictor = Predictor( # (1)!
    endpoint_id="<insert your endpoint ID here>",
    api_key="<insert your API key here>",
)

with Webcam(fps=0.5) as webcam:
    for frame in webcam: # (2)!
        frame.resize(width=512)
        frame.run_predict(predictor=predictor) # (3)!
        if "coffee-mug" in frame.predictions:
            frame.save_image(
                path="frame.png",
                include_predictions=True, # (4)!
            )
```

1. Creates the Predictor object with your API key and endpoint ID.
2. Iterate over each frame in the webcam feed.
3. Runs the inference on the frame.
4. Saves the frame with the predictions overlaid on top of the original image.
