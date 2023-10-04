Apart from counting and checking the predictions, we can also overlay the predictions on the original image. This is useful to see how well the model is performing. We can do this by using the `Frame.save_image` function. This function merges the original `Frame` image, the predictions, and the labels. It then overlays the predictions on top of the original image.

```python

predictor = Predictor(
    endpoint_id="<insert your endpoint ID here>",
    api_key="<insert your API key here>",
)
with Webcam(fps=0.5) as webcam:
    for frame in webcam:
        frame.resize(width=512)
        frame.run_predict(predictor=predictor)
        if "coffee-mug" in frame.predictions:
            frame.save_image(
                path="frame.png",
                include_predictions=True,
            )
```