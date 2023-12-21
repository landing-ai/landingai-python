A common task for computer vision is extracting text from images, also known as OCR (Optical Character Recognition).

The LandingLens Python SDK has OCR models available out-of-the-box, without the need to train your own model. The models are pre-trained on a variety of fonts types, and are optimized for accuracy and speed.

## Running OCR Inference

In order to extract text from an image, you can use the `landingai.predict.OcrPredictor` class, and run inference on a `Frame`.

The model works well with several font types. Let's try with this example image, which contains handwriting:

![Handwritten note that reads "Hi From Landing.AI"](../images/handwriting-hello.png)

```python
from landingai.predict import OcrPredictor
from landingai.pipeline.image_source import Frame

predictor = OcrPredictor(api_key="<insert your API key here>")  # (1)!

frame = Frame.from_image("/path/to/image.png")  # (2)!
frame.run_predict(predictor)  # (3)!

for prediction in frame.predictions:  # (4)!
    print(f"{prediction.text} (Confidence: {prediction.score})")  # (5)!
```

1. Create an `OcrPredictor` instance with your API key. Visit [https://app.landing.ai/](https://app.landing.ai/) and see [Getting the API Key](../getting-started#getting-the-api-key) for more details on how to get your API key.
2. Create a `Frame` instance from an image file. You could use any image source, such as a webcam, video file, screenshots, etc. See [Image Acquisition](../image-acquisition/image-acquisition.md) for more details.
3. Run inference on the frame to extract the text.
4. Iterate over the predictions.
5. Print the text and confidence score.

In the example above, the output should look like this:

```text
Hi From (Confidence: 0.9179285764694214)
Landing.AI (Confidence: 0.7755413055419922)
```

You can also use the `in` operator to check if a certain set of characters is present in the predictions:

```python
if "Landing" in frame.predictions:
    print("Found 'Landing' written in the text!")
```

The results may vary depending on the image quality, the font, and the language. Try with your own images to see how well the model performs, and [provide us feedback about the results](https://github.com/landing-ai/landingai-python/issues/new).
